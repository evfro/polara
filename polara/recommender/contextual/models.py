import numpy as np


class ItemPostFilteringMixin:
    def upvote_context_items(self, context, scores, test_users):
        if context is None:
            return

        userid = self.data.fields.userid
        itemid = self.data.fields.itemid
        context_data = self.data.context_data[context]
        try:
            upvote_items = context_data[userid].loc[test_users].map(context_data[itemid])
        except:
            print(f'Unable to upvote items in context "{context}"')
            return
        upvote_index = zip(*[(i, el) for i, l in enumerate(upvote_items) for el in l])

        context_idx_flat = np.ravel_multi_index(list(upvote_index), scores.shape)
        context_scores = scores.flat[context_idx_flat]

        upscored = scores.max() + context_scores + 1
        scores.flat[context_idx_flat] = upscored

    def upvote_relevant_items(self, scores, test_users):
        for context in self.data.context_data.keys():
            self.upvote_context_items(context, scores, test_users)

    def slice_recommendations(self, test_data, test_shape, start, stop, test_users):
        scores, slice_data = super().slice_recommendations(test_data, test_shape, start, stop, test_users)
        self.upvote_relevant_items(scores, test_users[start:stop])
        return scores, slice_data
