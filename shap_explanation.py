# %%
import time
import numpy as np
import scipy
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score


# %%
df1 = pd.read_csv('./data/your_obesity_data.csv', encoding='gbk') 
df1.head()

df2 = pd.read_csv('./data/your_standard_data.csv', encoding='gbk')
df2.head()


# %%
y1 = df1['Physical activity']
y1.head()

y2 = df2['Physical activity']
y2.head()


# %%

feature_order = ['Gender', 'Age', 'Land use diversity', 'Population density', 'Street Connectivity', 
                 'Road intersection density', 'Number of Bus Stops', 'Distance to Bus Stops', 
                 'Number of parks', 'Distance to park', 'Number of overpasses', 
                 'Streetscape Green Vision', 'Mean Tree Height', 'Mean NDVI', 'Mean LST', 
                 'Plants diversity', 'Total plants']
x1 = df1.drop('Physical activity', axis=1)[feature_order]
x1.head()

x2 = df2.drop('Physical activity', axis=1)[feature_order]
x2.head()

# %%
from sklearn.model_selection import train_test_split

seed = 42
xtrain1, xtest1, ytrain1, ytest1 = train_test_split(x1, y1, test_size=0.3, random_state=seed)

xtrain2, xtest2, ytrain2, ytest2 = train_test_split(x2, y2, test_size=0.3, random_state=seed)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import randint
from scipy import interpolate
import numpy as np
import pandas as pd
import shap  

with open('./models/OBESITY.pkl', 'rb') as file1: 
    obesity = pickle.load(file1)
    
with open('./models/STANDARD.pkl', 'rb') as file2: 
    standard = pickle.load(file2)


explainer1 = shap.TreeExplainer(obesity)  
shap_values1 = explainer1.shap_values(xtrain1)

explainer2 = shap.TreeExplainer(standard)  
shap_values2 = explainer2.shap_values(xtrain2)  


shap.summary_plot(shap_values1, xtrain1, feature_names=xtrain1.columns)  
shap.summary_plot(shap_values2, xtrain2, feature_names=xtrain1.columns)  


# %%
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


with open('./models/xgb_classifier_model1.pkl', 'rb') as file1: 
    obesity = pickle.load(file1)

with open('./models/xgb_classifier_model2.pkl', 'rb') as file2:  
    standard = pickle.load(file2)


importance_model1 = obesity.feature_importances_
importance_model2 = standard.feature_importances_


feature_names = obesity.get_booster().feature_names


importance_df1 = pd.DataFrame({'Feature': feature_names, 'Importance': importance_model1})
importance_df2 = pd.DataFrame({'Feature': feature_names, 'Importance': importance_model2})


importance_df1['Model'] = 'Obesity'
importance_df2['Model'] = 'Standard'


combined_importance = pd.concat([importance_df1, importance_df2])


combined_importance['Feature'] = pd.Categorical(combined_importance['Feature'], categories=feature_names, ordered=True)
combined_importance = combined_importance.sort_values('Feature')


custom_palette = {'Obesity': '#ff0052', 'Standard': '#0072ee'}


plt.figure(figsize=(12, 8))
barplot = sns.barplot(x='Importance', y='Feature', hue='Model', data=combined_importance, palette=custom_palette)


for p in barplot.patches:
    width = p.get_width()
    if width > 0:  
        barplot.annotate(f'{width:.4f}', 
                         (width, p.get_y() + p.get_height() / 2), 
                         ha='left', va='center', 
                         fontsize=10, color='black')

plt.title('Feature Importance Comparison', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.legend(title='Model', fontsize=12)
plt.tight_layout()
plt.show()


# %%

importance_model1 = obesity.feature_importances_
importance_model2 = standard.feature_importances_


feature_names = obesity.get_booster().feature_names


importance_df1 = pd.DataFrame({'Feature': feature_names, 'Importance': importance_model1})
importance_df2 = pd.DataFrame({'Feature': feature_names, 'Importance': importance_model2})
custom_palette = {'Obesity': '#ff0052', 'Standard': '#0072ee'}

plt.figure(figsize=(10, 6))
importance_df1 = importance_df1.sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=importance_df1, color = custom_palette['Obesity'])
# plt.title('Feature Importance for Obesity', fontsize=16)
plt.xlabel('Feature Importance for Obesity', fontsize=14)
plt.ylabel('Features', fontsize=14)


for p in plt.gca().patches:
    plt.annotate(f'{p.get_width():.4f}', 
                 (p.get_width(), p.get_y() + p.get_height() / 2), 
                 ha='left', va='center', 
                 fontsize=10, color='black')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
importance_df2 = importance_df2.sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=importance_df2, color = custom_palette['Standard'])
# plt.title('Feature Importance for Standard', fontsize=16)
plt.xlabel('Feature Importance for Standard', fontsize=14)
plt.ylabel('Features', fontsize=14)


for p in plt.gca().patches:
    plt.annotate(f'{p.get_width():.4f}', 
                 (p.get_width(), p.get_y() + p.get_height() / 2), 
                 ha='left', va='center', 
                 fontsize=10, color='black')
plt.tight_layout()
plt.show()

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager 

features = ['Gender', 'Age', 'Land use diversity', 'Population density', 'Street Connectivity', 
            'Road intersection density', 'Number of Bus Stops', 'Distance to Bus Stops', 'Number of parks', 
            'Distance to park', 'Number of overpasses', 'Streetscape Green Vision', 'Mean Tree Height', 
            'Mean NDVI', 'Mean LST', 'Plants diversity', 'Total plants']


font_properties = font_manager.FontProperties(family='Times New Roman', size=10)

for i in features:
    sns.set_theme(style="ticks", palette="deep", font_scale=1.3)
    fig = plt.figure(figsize=(6, 5), dpi=120)  
    ax = plt.subplot(111)


    x_values1 = xtrain1[i]
    shap_values1_i = shap_values1[:, xtrain1.columns.get_loc(i)]  

    x_values2 = xtrain2[i]
    shap_values2_i = shap_values2[:, xtrain2.columns.get_loc(i)]  


    df1 = pd.DataFrame({i: x_values1, 'SHAP': shap_values1_i})
    df2 = pd.DataFrame({i: x_values2, 'SHAP': shap_values2_i})


    mean_shap1 = df1.groupby(i)['SHAP'].mean().reset_index()
    mean_shap2 = df2.groupby(i)['SHAP'].mean().reset_index()


    plt.plot(mean_shap1[i], mean_shap1['SHAP'], color='#ff0052', alpha=0.5, label='Obesity Mean SHAP')
    plt.plot(mean_shap2[i], mean_shap2['SHAP'], color='#0072ee', alpha=0.5, label='Standard Mean SHAP')


    plt.scatter(x_values1, shap_values1_i, color='#ff0052', alpha=0.05, label='Obesity SHAP Points')  
    plt.scatter(x_values2, shap_values2_i, color='#0072ee', alpha=0.05, label='Standard SHAP Points')  


    sns.rugplot(data=xtrain1[[i]].sample(100), height=0.04, color='k', alpha=0.5)
    sns.rugplot(data=xtrain2[[i]].sample(100), height=0.04, color='b', alpha=0.5)


    plt.ylabel('Physical Activity', fontname="Times New Roman", fontsize=14)  
    plt.xlabel(f'{i}', fontname="Times New Roman", fontsize=14) 
    

    plt.xlim(min(xtrain1[i].min(), xtrain2[i].min()), max(xtrain1[i].max(), xtrain2[i].max()))


    plt.legend(framealpha=0.5, fontsize=10, prop=font_properties) 


    plt.tight_layout()  


    plt.savefig(f'./output/save_fig/shapplot_{i}.png', bbox_inches='tight') 
    plt.show()
