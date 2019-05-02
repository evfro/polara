from polara.recommender.data import RecommenderData


class ItemPostFilteringData(RecommenderData):
    def __init__(self, *args, item_context_mapping=None, **kwargs):
        super().__init__(*args, **kwargs)
        userid = self.fields.userid
        itemid = self.fields.itemid
        self.item_context_mapping = dict(**item_context_mapping)
        self.context_data = {context: dict.fromkeys([userid, itemid])
                                for context in item_context_mapping.keys()}

    def map_context_data(self, context):
        if context is None:
            return

        userid = self.fields.userid
        itemid = self.fields.itemid

        context_mapping = self.item_context_mapping[context]
        index_mapping = self.index.itemid.set_index('old').new
        mapped_index = {itemid: lambda x: x[itemid].map(index_mapping)}
        item_data = (context_mapping.loc[lambda x: x[itemid].isin(index_mapping.index)]
                                    .assign(**mapped_index)
                                    .groupby(context)[itemid]
                                    .apply(list))
        holdout = self.test.holdout
        try:
            user_data = holdout.set_index(userid)[context]
        except AttributeError:
            print(f'Unable to map {context}: holdout data is not recognized')
            return
        except KeyError:
            print(f'Unable to map {context}: not present in holdout')
            return
        # deal with mesmiatch between user and item data
        item_data = item_data.reindex(user_data.drop_duplicates().values, fill_value=[])

        self.context_data[context][userid] = user_data
        self.context_data[context][itemid] = item_data

    def update_contextual_data(self):
        holdout = self.test.holdout
        if holdout is not None:
            # assuming that for each user in holdout we have only 1 item
            assert holdout.shape[0] == holdout[self.fields.userid].nunique()

            for context in self.item_context_mapping.keys():
                self.map_context_data(context)

    def prepare(self, *args, **kwargs):
        super().prepare(*args, **kwargs)
        self.update_contextual_data()


    def set_test_data(self, *args, **kwargs):
        super().set_test_data(*args, **kwargs)
        self.update_contextual_data()
