import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PATH_TO_RESULT = r'C:\Users\quang\PycharmProjects\DL_NLP_TUH\NLP\NER\NER_Error_analysis\Model results\result_scores - Copy.csv'
MODEL_TO_VISUALIZE = 'phobert'  # or 'bilstm'
##
df = pd.read_csv(PATH_TO_RESULT)
df_grouped_by_model = df.groupby(by=['model'])

if MODEL_TO_VISUALIZE == 'phobert':
    batch_sizes = ['2', '4', '8', '16', '32']
elif MODEL_TO_VISUALIZE == 'bilstm':
    batch_sizes = ['1', '36']

model_dict = df_grouped_by_model['test_f1'].apply(list).to_dict()
model_names = list(model_dict.keys())

model_list = []
for model in model_names:
    temp_result_dict = {}
    for i, size in enumerate(batch_sizes):
        temp_result_dict[size] = model_dict[model][i]
    model_list.append(temp_result_dict)

##
df_max_scores = pd.DataFrame(columns=batch_sizes)
for report in model_list:
    df_score = pd.DataFrame(report, index=[0])
    df_max_scores = pd.concat((df_max_scores, df_score), axis=0, ignore_index=True)

df_final = pd.concat([df_max_scores, pd.DataFrame({'model': model_names})], axis=1)

sns.set_theme(style="white")
# Draw a nested barplot by species and sex
if MODEL_TO_VISUALIZE == 'phobert':
    g = sns.catplot(data=df_final.melt(id_vars='model', var_name="quantity"), kind="bar", x="model", y="value", hue="quantity", ci="sd", aspect=2.7, height=4)
elif MODEL_TO_VISUALIZE == 'bilstm':
    g = sns.catplot(data=df_final.melt(id_vars='model', var_name="quantity"), kind="bar", x="model", y="value", hue="quantity", ci="sd", aspect=0.44, height=4)

g.set(ylim=(0, 1))
ax = g.facet_axis(0, 0)
for i, p in enumerate(ax.patches):
    if p.get_height() == 0:
        continue
    if MODEL_TO_VISUALIZE == 'phobert' and p.get_height() in [0.92501, 0.93801, 0.94201, 0.94501]:
        ax.text(p.get_x() - 0.002, p.get_height() * 1.02, '{0:.3f}*'.format(p.get_height()), color='black', rotation='horizontal', size=7)
    if MODEL_TO_VISUALIZE == 'bilstm' and p.get_height() in [0.906]:
        ax.text(p.get_x() - 0.002, p.get_height() * 1.02, '{0:.3f}*'.format(p.get_height()), color='black', rotation='horizontal', size=7)
    else:
        ax.text(p.get_x() - 0.002, p.get_height() * 1.02, '{0:.3f}'.format(p.get_height()), color='black', rotation='horizontal', size=7)

g.despine(left=True)
g.set_axis_labels("", "value")
g.legend.set_title("")
# plt.show()
plt.savefig(r'C:\Users\quang\PycharmProjects\DL_NLP_TUH\NLP\NER\NER_Error_analysis\Output\model_results.png', bbox_inches='tight', dpi=250)  ##
