import pandas as pd

from polara.recommender.data import RecommenderData


def define_state(usn, hsz, trt):
    if usn:
        if (hsz > 0) and (trt > 0):
            state = 4
        else:
            raise ValueError('Invalid parameters')
    else:
        if hsz == 0:
            if trt == 0:
                state = 1
            else:
                state = 11
        else:  # hsz > 0
            if trt == 0:
                state = 2
            else:  # trt > 0
                state = 3
    return state


def assign_config(usn, hsz, trt, state):
    data_model._test_ratio = trt
    data_model._holdout_size = hsz
    data_model._warm_start = usn
    data_model._state = state


fields = ['userid', 'itemid', 'rating']
data = pd.DataFrame(columns=fields)
data_model = RecommenderData(data, *fields)

for usn in [False, True]:
    for hsz in [0, 0.25, 0.5, 1, 3]:
        for trt in [0, 0.1, 0.2]:
            try:
                state = define_state(usn, hsz, trt)
            except ValueError as e:
                print('{}: usn {}, hsz {}, trt {}\n'.format(e, usn, hsz, trt))
                continue
            print('current config: usn - {}, hsz - {}, trt - {}, state - {}'.format(usn, hsz, trt, state))
            assign_config(usn, hsz, trt, state)

            for usn_new in [False, True]:
                for hsz_new in [0, 0.25, 1]:
                    for trt_new in [0, 0.1]:
                        print('usn: {:b}, hsz: {:4}, trt: {:4}'.format(usn_new, hsz_new, trt_new))
                        data_model.test_ratio = trt_new
                        data_model.holdout_size = hsz_new
                        data_model.warm_start = usn_new
                        try:
                            data_model._validate_config()
                        except ValueError as e:
                            print(e)
                        else:
                            new_stt, upd = data_model._check_state_transition()
                            print('new state: {:3}, update rule: {}     '.format(new_stt, upd),)
                            print(list(data_model._change_properties))

                        assign_config(usn, hsz, trt, state)
                        data_model._change_properties.clear()
            print()
