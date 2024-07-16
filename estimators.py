"""
    LMKG: Learned Models for Cardinality Estimation in Knowledge Graphs

    Source Code used as is or modified from the above mentioned source
"""

import time
import numpy as np
import torch
import made

class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):

        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))

class BaseDistributionEstmationBatch(CardEst):
    '''
        Estimation from the AR model without progressive sampling.
    '''

    def __init__(
            self,
            model,
            table,
            r,
            device=None,
            seed=False,
            cardinality=None,
            shortcircuit=False,
            num_sample=1000,
            mapping_for_grid_cell=None
    ):
        super(BaseDistributionEstmationBatch, self).__init__()
        torch.set_grad_enabled(False)
        self.model = model
        self.table = table
        self.shortcircuit = True

        self.num_samples = num_sample

        self.seed = seed
        self.device = device

        self.cardinality = cardinality
        if cardinality is None:
            self.cardinality = table.cardinality

        with torch.no_grad():
            self.init_logits = self.model(
                torch.zeros(1, self.model.nin, device=device))

        # Inference optimizations below.
        self.traced_fwd = None
        # We can't seem to trace this because it depends on a scalar input.
        self.traced_encode_input = model.EncodeInput

        self.mapping_for_grid_cell = mapping_for_grid_cell

        if 'MADE' in str(model):
            for layer in model.net:
                if type(layer) == made.MaskedLinear:
                    if layer.masked_weight is None:
                        layer.masked_weight = layer.mask * layer.weight
                        print('Setting masked_weight in MADE, do not retrain!')

        for p in model.parameters():
            p.detach_()
            p.requires_grad = False
        self.init_logits.detach_()

        with torch.no_grad():
            self.kZeros = torch.zeros(self.num_samples,
                                      self.model.nin,
                                      device=self.device)

            self.inp = self.traced_encode_input(self.kZeros)

            self.inp = self.inp.view(self.num_samples, -1)

    def _sample_n_batch(self,
                        num_samples,
                        ordering,
                        columns,
                        operators,
                        vals,
                        inp=None,
                        num_data_in_batch=-1):

        ncols = len(columns)

        inp = self.inp[:num_data_in_batch]

        original_vals = vals.copy()
        '''
            Do the encoding for the wildcards and add them to the array represent the query ('vals') 
        '''
        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]
            if operators[natural_idx] is None:
                '''encoding for the wildcards'''
                if natural_idx == 0:
                    self.model.EncodeInput(
                        None,
                        natural_col=0,
                        out=inp[:, :self.model.
                            input_bins_encoded_cumsum[0]])
                else:
                    l = self.model.input_bins_encoded_cumsum[natural_idx -
                                                             1]
                    r = self.model.input_bins_encoded_cumsum[natural_idx]
                    self.model.EncodeInput(None,
                                           natural_col=natural_idx,
                                           out=inp[:, l:r])
            else:
                # from the batch of samples take only the values for the current column
                discretized_vals_for_col = [[tmp_list[natural_idx]] for tmp_list in original_vals]
                # put them in the required format
                data_to_encode = torch.LongTensor(discretized_vals_for_col)
                if natural_idx == 0:
                    self.model.EncodeInput(
                        data_to_encode.to(self.device),
                        natural_col=0,
                        out=inp[:, :self.model.
                            input_bins_encoded_cumsum[0]])
                else:
                    # starting bit postion for current column
                    l = self.model.input_bins_encoded_cumsum[natural_idx - 1]
                    # ending bit postion for current column
                    r = self.model.input_bins_encoded_cumsum[natural_idx]
                    self.model.EncodeInput(data_to_encode.to(self.device),
                                           natural_col=natural_idx,
                                           out=inp[:, l:r])

        # create the logits for the encoded query
        logits = self.model.forward_with_encoded_input(inp)

        # p_x_1 = np.ones(len(vals)).reshape(-1, 1)
        p_x_1 = torch.ones(len(vals), device=self.device).reshape(-1, 1)

        for i in range(ncols):
            natural_idx = i if ordering is None else ordering[i]

            if operators[natural_idx] is not None:
                '''this means that the column is not a wildcard'''
                # take the probabilities for the column of interest
                probs_i = torch.softmax(
                    self.model.logits_for_col(natural_idx, logits), 1)

                # since we have a fixed value for the columns, take only the probabilities for those columnns
                # needed_probs_i = np.take_along_axis(probs_i.cpu().numpy(), original_vals[:, natural_idx].reshape(-1, 1).astype(int),
                #                                     axis=1)

                needed_probs_i = torch.gather(probs_i, 1,
                                              torch.tensor(original_vals[:, natural_idx].reshape(-1, 1).astype(int),
                                                           device=self.device))
                p_x_1 *= needed_probs_i

        return p_x_1

    def QueryWithCompressionBatch(self, columns, operators, vals, num_data_in_batch=-1):
        ordering = self.model.orderings[0]

        """
            Transfer the values 
        """
        final_values_batch = list()

        for singel_val in vals:
            final_values = list()
            final_columns = list()
            final_signs = list()

            modified_columns_index = 0
            for val_indx, val in enumerate(singel_val):
                if val_indx in self.model.compressor_element.split_columns_index:
                    # every column at the beginning will be split into 2 columns
                    how_many_times_compressed = 2

                    if val is not None:
                        quotient, reminder = self.model.compressor_element.split_single_value_for_column(val, val_indx)
                        # save the reminder for the future
                        all_reminders = list()
                        all_reminders.append(reminder)

                        # split the column into the required number of columns
                        while how_many_times_compressed < self.model.compressor_element.root:
                            # get the quotient and reminder from the quotient in the previous iteration
                            quotient, reminder = self.model.compressor_element.split_single_value_for_column(
                                int(quotient),
                                val_indx)

                            # save the reminder, it will represent a separate column
                            all_reminders.append(reminder)
                            how_many_times_compressed += 1

                        # store the information for the quotient as a separate column
                        final_columns.append(self.model.compressor_element.model_column_names[modified_columns_index])
                        modified_columns_index += 1
                        final_values.append(int(quotient))
                        # store the sign
                        final_signs.append(operators[val_indx])

                        # iterate over the reminders in a reversed order such that the last reminder
                        # is actually the reminder for the quotient the one after that is the reminder for the number
                        # made by the quotient and the first reminder, etc...
                        for num_rem, rem_val in enumerate(reversed(all_reminders)):
                            # store the id of the column
                            final_columns.append(
                                self.model.compressor_element.model_column_names[modified_columns_index])

                            modified_columns_index += 1
                            # store the value of the column
                            final_values.append(int(rem_val))

                            # store the sign
                            final_signs.append(operators[val_indx])
                    else:
                        """if the value is none just take the column into consideration everything else is none"""
                        for num in range(how_many_times_compressed):
                            final_columns.append(
                                self.model.compressor_element.model_column_names[modified_columns_index])
                            final_signs.append(None)
                            final_values.append(None)
                            modified_columns_index += 1

                else:
                    final_columns.append(self.model.compressor_element.model_column_names[modified_columns_index])
                    final_signs.append(operators[val_indx])
                    final_values.append(val)

                    modified_columns_index += 1

            final_values_batch.append(final_values)

        inv_ordering = [None] * len(self.model.compressor_element.model_column_names)
        for natural_idx in range(len(self.model.compressor_element.model_column_names)):
            inv_ordering[ordering[natural_idx]] = natural_idx

        with torch.no_grad():
            p = self._sample_n_batch(
                self.num_samples,
                inv_ordering,
                np.array(final_columns),
                np.array(final_signs),
                np.array(final_values_batch),
                inp=self.inp.zero_(),
                num_data_in_batch=num_data_in_batch)

            # result = np.round(p * self.cardinality).astype(dtype=np.int32,
            #                                                copy=False)
            # return result
            return torch.round(p * self.cardinality)
