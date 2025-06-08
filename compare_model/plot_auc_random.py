import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
plt.rcParams['font.size'] = 14

# 1. 读取 7 个 CSV 文件

df1_1 = pd.read_csv('./KFLM2/auc_results_BACE_random.csv')
df1_2 = pd.read_csv('./KFLM2/auc_results_BBBP_random.csv')
df1_3 = pd.read_csv('./KFLM2/auc_results_ClinTox_random.csv')
df1_4 = pd.read_csv('./KFLM2/auc_results_sider_random.csv')
df1_5 = pd.read_csv('./KFLM2/auc_results_Tox21_random.csv')

data_list = [df1_1['AUC'],  df1_2['AUC'], df1_3['AUC'], df1_4['AUC'],  df1_5['AUC']]


df2_1 = pd.read_csv('CoMPT/auc_results_bace_random.csv')
df2_2 = pd.read_csv('CoMPT/auc_results_bbbp_random.csv')
df2_3 = pd.read_csv('CoMPT/auc_results_clintox_random.csv')
df2_4 = pd.read_csv('CoMPT/auc_results_sider_random.csv')
df2_5 = pd.read_csv('CoMPT/auc_results_tox21_random.csv')

data_list2 = [df2_1['auc'],  df2_2['auc'], df2_3['auc'], df2_4['auc'], df2_5['auc']]

df3_1 = pd.read_csv('MoleSG/auc_results_bace_random.csv')
df3_2 = pd.read_csv('MoleSG/auc_results_bbbp_random.csv')
df3_3 = pd.read_csv('MoleSG/auc_results_clintox_random.csv')
df3_4 = pd.read_csv('MoleSG/auc_results_sider_random.csv')
df3_5 = pd.read_csv('MoleSG/auc_results_tox21_random.csv')

data_list3 = [df3_1['auc'],  df3_2['auc'], df3_3['auc'], df3_4['auc'], df3_5['auc']]


# 2. 自定义 xticks 的标签
xticks_labels = ['BACE', 'BBBP', 'ClinTox', 'SIDER', 'Tox21' ]
data = pd.DataFrame(dict(zip(xticks_labels, data_list)))
data2 = pd.DataFrame(dict(zip(xticks_labels, data_list2)))
data3 = pd.DataFrame(dict(zip(xticks_labels, data_list3)))


data2['Group'] = 'CoMPT'
data['Group'] = 'KFLM2'
data3['Group'] = 'MoleSG'

combined_data = pd.concat([
    data2.melt(id_vars='Group', var_name='Dataset', value_name='AUC'),
    data3.melt(id_vars='Group', var_name='Dataset', value_name='AUC'),
    data.melt(id_vars='Group', var_name='Dataset', value_name='AUC')
])

print(combined_data)

custom_palette = {'SAMPL': '#55B7E6', 'Llinas': '#F09148', 'Delaney': '#FF9896', 'Lipophilicity': '#DBDB8D', 'pKa': '#C59D94',}
custom_palette = {'Random': '#55B7E6', 'Scaffold': '#F09148'}
# 绘制箱线图
plt.figure(figsize=(6, 4.0))
#sns.boxplot(x='Model', y='Pearson', hue='Group', data=combined_data, palette=custom_palette)
sns.boxplot(x='Dataset', y='AUC', hue='Group', data=combined_data, palette='Set2', fliersize=3,)
#sns.violinplot(data=combined_data, palette="Set3", cut=0, linewidth=1, )
#ax = sns.violinplot(x='Dataset', y='AUC', hue='Group', data=combined_data, 
#                   palette="Set2", split=False, inner="quartile", linewidth=1, cut=0)

plt.title('Random Splitting', fontsize=14)
plt.xlabel('Dataset')
#plt.legend(title='Dataset', fontsize=10)
plt.legend(fontsize=10)
#plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(len(xticks_labels)), xticks_labels,  fontsize=12)
# 5. 添加标题和标签
#plt.title('Violin Plot of 7 CSV Files')
#plt.xlabel('Methods')
plt.ylabel('AUC')

#plt.title(label='Llinas', fontsize=12)
# 6. 调整布局
plt.tight_layout()
plt.savefig("prediction_auc_methond_random_compare.png", dpi=300, bbox_inches='tight')

# 7. 显示图形
plt.show()
