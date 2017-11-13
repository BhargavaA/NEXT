import next.utils as utils
from next.apps.AppDashboard import AppDashboard
import pandas as pd
import numpy as np


class MyAppDashboard(AppDashboard):
    def __init__(self,db,ell):
        AppDashboard.__init__(self,db,ell)

    def cumulative_reward_plot(self, app, butler):
        """
        Description: Returns multiline plot where there is a one-to-one mapping lines to
        algorithms and each line indicates the error on the validation set with respect to number of reported answers

        Expected input:
          None

        Expected output (in dict):
          (dict) MPLD3 plot dictionary
        """
        # get list of algorithms associated with project
        # utils.debug_print('came into Dashboard')
        args = butler.experiment.get(key='args')
        # num_algs = len(args['alg_list'])
        # utils.debug_print('num_tries: ', args['num_tries'])
        # alg_labels = []
        # for i in range(num_algs):
        #     alg_labels += [args['alg_list'][i]['alg_label']]


        plot_data = butler.experiment.get(key='plot_data')
        # utils.debug_print('butler.algs.plot_data in Dashboard: ', plot_data)
        df = pd.DataFrame(plot_data)
        df.columns = [u'alg', u'arm_pulled', u'initial_arm', u'participant_uid', u'rewards', u'time']
        # utils.debug_print('df: ', df)
        # df = df.pivot_table(columns='initial arm', index='time', values='rewards', aggfunc=np.mean)
        # utils.debug_print('df: ', df)
        # utils.debug_print('Came into Dashbord, trying to print algs and init_arms')
        algs = list(df['alg'].unique())
        utils.debug_print('algs: ', algs)
        init_arms = df['initial_arm'].unique()
        utils.debug_print('init_arms: ', init_arms)
        import matplotlib.pyplot as plt
        import mpld3
        fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(axisbg='#EEEEEE'))

        # T = args['num_tries']
        # utils.debug_print('T: ', T)

        for alg in algs:
            alg_results = np.zeros(T)
            for i, init_arm in enumerate(init_arms):
                print alg
                print init_arm
                result = df.query('alg == "{alg}" and initial_arm == {iarm}'.format(alg=alg, iarm=init_arm))[
                    ['time', 'rewards', 'participant_uid']].groupby('time').mean()
                rewards = np.array(result['rewards'])
                # utils.debug_print('rewards: ', rewards)
                # utils.debug_print('len rewards: ', len(rewards))
                # utils.debug_print('alg_results: ', alg_results)
                alg_results[0:len(rewards)] += rewards / float(len(init_arms))

            ax.plot(range(len(rewards)), np.cumsum(rewards), label='{alg}'.format(alg=alg))
            ax.set_xlabel('Time')
            ax.set_ylabel('Average cumulative rewards')

        ax.set_title('Cumulative rewards', size=10)
        legend = ax.legend(loc=2, ncol=2, mode="expand")
        for label in legend.get_texts():
            label.set_fontsize('xx-small')

        plot_dict = mpld3.fig_to_dict(fig)
        plt.close()
        return plot_dict
