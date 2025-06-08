import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
plt.rcParams['font.size'] = 14

# 1. 读取 7 个 CSV 文件

df1_1 = pd.read_csv('./KFLM2/pearson_results_SAMPL_scaffold.csv')
df1_2 = pd.read_csv('./KFLM2/pearson_results_Delaney_scaffold.csv')
df1_3 = pd.read_csv('./KFLM2/pearson_results_Lipophilicity_scaffold.csv')
df1_4 = pd.read_csv('./KFLM2/pearson_results_Warrior_scaffold.csv')

data_list = [df1_1['RMSE'],  df1_2['RMSE'], df1_3['RMSE'], df1_4['RMSE'],]


df2_1 = pd.read_csv('CoMPT/rmse_results_freesolv_scaffold.csv')
df2_2 = pd.read_csv('CoMPT/rmse_results_esol_scaffold.csv')
df2_3 = pd.read_csv('CoMPT/rmse_results_lipophilicity_scaffold.csv')
df2_4 = pd.read_csv('CoMPT/rmse_results_datawarrior_scaffold.csv')

data_list2 = [df2_1['rmse'],  df2_2['rmse'], df2_3['rmse'], df2_4['rmse'],]

df3_1 = pd.read_csv('MoleSG/rmse_results_freesolv_scaffold.csv')
df3_2 = pd.read_csv('MoleSG/rmse_results_esol_scaffold.csv')
df3_3 = pd.read_csv('MoleSG/rmse_results_lipophilicity_scaffold.csv')
df3_4 = pd.read_csv('MoleSG/rmse_results_datawarrior_scaffold.csv')

data_list3 = [df3_1['rmse'],  df3_2['rmse'], df3_3['rmse'], df3_4['rmse'],]


# 2. 自定义 xticks 的标签
xticks_labels = ['SAMPL', 'Delaney', 'Lipo', 'pKa' ]
#xticks_labels = ['CoMPT', 'MoleSG', 'KFLM2']
data = pd.DataFrame(dict(zip(xticks_labels, data_list)))
data2 = pd.DataFrame(dict(zip(xticks_labels, data_list2)))
data3 = pd.DataFrame(dict(zip(xticks_labels, data_list3)))

print(data)

data2['Group'] = 'CoMPT'
data['Group'] = 'KFLM2'
data3['Group'] = 'MoleSG'

# 合并数据
combined_data = pd.concat([data2.melt(id_vars='Group', var_name='Dataset', value_name='RMSE'),
                           data3.melt(id_vars='Group', var_name='Dataset', value_name='RMSE'),
                           data.melt(id_vars='Group', var_name='Dataset', value_name='RMSE'),])

print(combined_data)
custom_palette = {'SAMPL': '#55B7E6', 'Llinas': '#F09148', 'Delaney': '#FF9896', 'Lipophilicity': '#DBDB8D', 'pKa': '#C59D94',}
custom_palette = {'Random': '#55B7E6', 'Scaffold': '#F09148'}
# 绘制箱线图
plt.figure(figsize=(5, 4.0))
#sns.boxplot(x='Dataset', y='RMSE', hue='Group', data=combined_data, palette=custom_palette)
sns.boxplot(x='Group', y='RMSE', hue='Dataset', data=combined_data, palette='Set2', fliersize=3,)
plt.title('Scaffold Split', fontsize=12)
plt.ylim([0, 4])
#plt.xlabel('Dataset')
#plt.legend(title='Dataset', fontsize=10)
plt.legend(fontsize=10, loc='best')
#plt.grid(True, linestyle='--', alpha=0.6)
#plt.xticks(range(len(xticks_labels)), xticks_labels,  fontsize=12)
# 5. 添加标题和标签
#plt.title('Violin Plot of 7 CSV Files')
plt.xlabel('Methods')
plt.ylabel('RMSE')

#plt.title(label='Llinas', fontsize=12)
# 6. 调整布局
plt.tight_layout()
plt.savefig("methond_scaffold_compare.png", dpi=300)

# 7. 显示图形
plt.show()
