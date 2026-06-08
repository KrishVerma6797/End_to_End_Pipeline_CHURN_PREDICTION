import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV 
from xgboost import XGBClassifier




#loading data

df=pd.read_csv('churn.csv')

#understanding the data
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
df.drop('customerID',axis=1,inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace=True)


# #EDA
sns.countplot(x='Churn',data=df)
plt.show()

sns.countplot(x='Contract',hue='Churn',data=df)
plt.show()

sns.boxplot(x='Churn',y='tenure',data=df)
plt.show()

sns.boxplot(x='Churn',y='MonthlyCharges',data=df)
plt.show()


#encoding 
cat_col=df.drop("Churn",axis=1).select_dtypes(include='object').columns
encoders={}
for col in cat_col:
    le=LabelEncoder() 
    df[col]=le.fit_transform(df[col])
    encoders[col]=le
le_churn = LabelEncoder()
df['Churn'] = le_churn.fit_transform(df['Churn'])

#correlation
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.show()


#feature Selection
selected_features = [
    'TotalCharges',
    'MonthlyCharges',
    'tenure',
    'Contract',
    'PaymentMethod',
    'OnlineSecurity',
    'TechSupport'
]
y=df['Churn']
x = df[selected_features]

#model trainig
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


#Logistic Regression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print("Logistic Regression")
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))



#random forest
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)
xgb.fit(x_train,y_train)
y_pred_xgb=xgb.predict(x_test)
print("XGBoost")
print('Accuracy:', accuracy_score(y_test, y_pred_xgb))
print('Classification Report:\n', classification_report(y_test, y_pred_xgb))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_xgb))

#feature_importance
feature_importance=pd.DataFrame({
    'Features':x.columns,
    'Importances':xgb.feature_importances_
})
feature_importance=feature_importance.sort_values(by='Importances',ascending=False)
print(feature_importance)
feature_importance.to_csv('feature_importance.csv',index=False)

#plots
plt.figure(figsize=(10,6))
sns.barplot(x='Importances',y='Features',data=feature_importance)
plt.title('Feature Importances')    
plt.show()  

#parameters tuning
param_grid = {
    'n_estimators': [100,200,300,500],
    'max_depth': [3,4,5,6,8],
    'learning_rate': [0.01,0.05,0.1,0.2],
    'subsample': [0.8,0.9,1.0],
    'colsample_bytree': [0.8,0.9,1.0],
    'gamma': [0,0.1,0.2,0.3]
}
xgb_random = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=20,
    scoring='f1',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
xgb_random.fit(x_train,y_train)
best_model=xgb_random.best_estimator_
print("Best Parameters:", xgb_random.best_params_)
y_pred_xgb_random=best_model.predict(x_test)

print("XGBoost Tuned")
print(accuracy_score(y_test,y_pred_xgb_random))
print(classification_report(y_test,y_pred_xgb_random))
print(feature_importance.head(10))
#Comparing
results=pd.DataFrame({
    'Model':['Logistic Regression','XGBoost','XGBoost(Tuned)'],
    'Accuracy':[accuracy_score( y_test,y_pred),accuracy_score(y_test, y_pred_xgb),accuracy_score(y_test, y_pred_xgb_random)],
    'Classification Report':[classification_report(y_test, y_pred),classification_report(y_test, y_pred_xgb),classification_report(y_test, y_pred_xgb_random)],
    'Confusion Matrix':[confusion_matrix(y_test, y_pred),confusion_matrix(y_test, y_pred_xgb),confusion_matrix(y_test, y_pred_xgb_random)]   
})
print(results)
#saving model

joblib.dump(encoders,'encoders.pkl')
joblib.dump(best_model,"model.pkl") 
joblib.dump(scaler,"scaler.pkl")
x_test_df = pd.DataFrame(
    x_test,
    columns=x.columns
)




from sklearn.metrics import roc_curve, roc_auc_score

y_prob = best_model.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost")
plt.legend()
plt.show()
print("ROC-AUC Score:", auc_score)

from sklearn.metrics import roc_auc_score

lr_auc = roc_auc_score(y_test, lr.predict_proba(x_test)[:,1])
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(x_test)[:,1])
tuned_auc = roc_auc_score(y_test, best_model.predict_proba(x_test)[:,1])

print("LR ROC-AUC:", lr_auc)
print("XGB ROC-AUC:", xgb_auc)
print("Tuned XGB ROC-AUC:", tuned_auc)
