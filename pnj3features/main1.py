# Perform several tasks:

# 1: classify datasets
# 2: get results tables
# 3: plot all months magnitudes
# 4: t-student tests


steps = [1]

period_vec = ['jra', 'piControl', 'abrupt-4xCO2', 'piControl', 'abrupt-4xCO2',
              'piControl', 'abrupt-4xCO2']
model_vec = ['jra', 'CESM2-WACCM', 'CESM2-WACCM', 'CanESM5', 'CanESM5',
             'IPSL-CM6A-LR', 'IPSL-CM6A-LR']

level = '010'

mainpath = 'TBD'

pathnc = mainpath + 'results/ncremapped/'
pathh5 = mainpath + 'datos/h5remapped/'
pathclass = './results/class/'
pathplot = './plots/'
pathclassifier = 'TBD'
pathtables = './results/tables/'


def main():
    for step in steps:

        if step == 1:
            from pnj3f.pnj_classify import pnj_classify

            for i in range(len(model_vec)):
                pnj_classify(model_vec[i], period_vec[i], level, pathh5, pathplot, pathclassifier, pathclass)

        if step == 2:
            from pnj3f.global_tables import tablesh5
            tablesh5(pathclass, pathtables, level)

        if step == 3:
            from pnj3f.plotmisc2 import plot_months
            for i in range(len(model_vec)):
                plot_months(model_vec[i], period_vec[i], level, pathh5, pathclass, pathplot)

        if step == 4:
            from pnj3f.stat_changes import changesh5
            changesh5()


if __name__ == "__main__":
    main()
