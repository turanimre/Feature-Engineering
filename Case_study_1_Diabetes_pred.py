import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


#### İş Problemi

'''
Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
gerçekleştirmeniz beklenmektedir.
'''


#### Veri seti & Değişkenler

'''
Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
yapılan diyabet araştırması için kullanılan verilerdir.
Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.


diabetes.csv

Pregnancies:  Hamilelik sayısı
Glucose:  Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
Blood:  Pressure Kan Basıncı (Küçük tansiyon) (mm Hg)
SkinThickness:  Cilt Kalınlığı
Insulin:  2 saatlik serum insülini (mu U/ml)
DiabetesPedigreeFunction:  Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
BMI:  Vücut kitle endeksi
Age:  Yaş (yıl)
Outcome:  Hastalığa sahip (1) ya da değil (0)

'''



#######################
## Görev 1: Keşifçi Veri Analizi
#######################

df = pd.read_csv("Miuul_Course_1/Feature-Engineering/Dataset/diabetes.csv")

### Adım 1: Genel resmi inceleyiniz.

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

### Adım 2: Numerik ve kategorik değişkenleri yakalayınız

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df, 10, 20)

### Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

#### Kategorik değişkenlerin analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Outcome", True)

#### Numerik değişkenlerin anazlizi.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for i in num_cols:
    num_summary(df, i)


### Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

#### Numeric değişkenlere göre target değişken analizi

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for i in num_cols:
    target_summary_with_num(df, "Outcome", i)


### Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(check_outlier(df, col))


### Adım 6: Eksik gözlem analizi yapınız.

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)


### Adım 7: Korelasyon analizi yapınız.


sns.heatmap(df.corr())


#######################
## Görev 2: Feature Engineering
#######################

### Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.


#### Eksik değerlerin ataması Glikoz vb. gibi değişkenler 0 olamayacağı için bunları nan olarak kabul edeceğiz.

cols = [col for col in num_cols if col != "Pregnancies"]

for col in cols:
    df.loc[df[col] == 0, col] = np.nan

na_columns = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_columns)

#### Eksik değerlerin ataması

for col in na_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

#### Aykırı değerlerin bulunması ve baskılanması

def replace_with_thresholds(dataframe, variable, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


for col in num_cols:
    print(check_outlier(df, col))


### Adım 2: Yeni değişkenler oluşturunuz.


#### Age değişkeninden yeni bir kategorik değişken oluşturuyoruz.

df.groupby("Age").agg({"Age": "count"}) # Genel yaşdağılımının nasıl olduğunu anlamaya çalışıyoruz
df["Age"].describe().T # Genel yaşdağılımının nasıl olduğunu anlamaya çalışıyoruz


df.loc[(df["Age"] >= 21) & (df["Age"] <= 25), "Age_cat"] = "Young"
df.loc[(df["Age"] > 25) & (df["Age"] <= 50), "Age_cat"] = "Mature"
df.loc[(df["Age"] > 50), "Age_cat"] = "Senior"

#### Glikoz değerlerine göre 2 saatte, 140 mg/dL veya daha düşük kan şekeri seviyesi normal kabul edilir, 140 ila 199 mg/dL prediyabetiniz olduğunu ve 200 mg/dL veya daha yüksek olması diyabetiniz olduğunu gösterir.

df["Glucose"].describe().T

df.loc[df["Glucose"] <= 140, "glucose_cat"] = "Normal"
df.loc[(df["Glucose"] > 140) & (df["Glucose"] < 200), "glucose_cat"] = "Prediabetes" # Max  değerimiz df["Glucose"].max() 199 olduğu için böyle brakacağız.

#### BMI değerine göre zayıf, sağlıklı vb. gibi çıkarımlarda bulunma.

df["BMI"].describe([0.1]).T

df.loc[(df["BMI"] < 18.5), "BMI_cat"] = "Underweight"
df.loc[(df["BMI"] >= 18) & (df["BMI"] < 25), "BMI_cat"] = "Healthy"
df.loc[(df["BMI"] >= 25) & (df["BMI"] < 30), "BMI_cat"] = "Overweight"
df.loc[(df["BMI"] >= 30), "BMI_cat"] = "Obese"


## Yaş ve beden kitle indeksini bir arada düşünerek
# df.loc[(df["BMI"] >= 30) & (df["Age"] > 50)]


df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] <= 25)), "Age_BMI_cat"] = "UnderweightYoung"
df.loc[(df["BMI"] < 18.5) & ((df["Age"] > 25) & (df["Age"] <= 50)), "Age_BMI_cat"] = "UnderweightMature"
df.loc[((df["BMI"] >= 18) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] <= 25)), "Age_BMI_cat"] = "HealthyYoung"
df.loc[((df["BMI"] >= 18) & (df["BMI"] < 25)) & ((df["Age"] > 25) & (df["Age"] <= 50)), "Age_BMI_cat"] = "HealthyMature"
df.loc[((df["BMI"] >= 18) & (df["BMI"] < 25)) & (df["Age"] > 50), "Age_BMI_cat"] = "HealthySenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] <= 25)), "Age_BMI_cat"] = "OverweightYoung"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] > 25) & (df["Age"] <= 50)), "Age_BMI_cat"] = "OverweightMature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] > 50), "Age_BMI_cat"] = "OverweightSenior"
df.loc[(df["BMI"] >= 30) & ((df["Age"] >= 21) & (df["Age"] <= 25)), "Age_BMI_cat"] = "ObeseYoung"
df.loc[(df["BMI"] >= 30) & ((df["Age"] > 25) & (df["Age"] <= 50)), "Age_BMI_cat"] = "ObeseMature"
df.loc[(df["BMI"] >= 30) & (df["Age"] > 50), "Age_BMI_cat"] = "ObeseSenior"


# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
# df.loc[((df["Glucose"] > 140) & (df["Glucose"] < 200)) & (df["Age"] > 50)]

df.loc[(df["Glucose"] <= 140) & ((df["Age"] >= 21) & (df["Age"] <= 25)), "Age_Glucose_cat"] = "NormalYoung"
df.loc[(df["Glucose"] <= 140) & ((df["Age"] > 25) & (df["Age"] <= 50)), "Age_Glucose_cat"] = "NormalMature"
df.loc[(df["Glucose"] <= 140) & (df["Age"] > 50), "Age_Glucose_cat"] = "NormalSenior"
df.loc[((df["Glucose"] > 140) & (df["Glucose"] < 200)) & ((df["Age"] >= 21) & (df["Age"] <= 25)), "Age_Glucose_cat"] = "PrediabetesYoung"
df.loc[((df["Glucose"] > 140) & (df["Glucose"] < 200)) & ((df["Age"] > 25) & (df["Age"] <= 50)), "Age_Glucose_cat"] = "PrediabetesMature"
df.loc[((df["Glucose"] > 140) & (df["Glucose"] < 200)) & (df["Age"] > 50), "Age_Glucose_cat"] = "PrediabetesSenior"


### Adım 3: Encoding işlemlerini gerçekleştiriniz.

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=15)
target = "Outcome"
cat_cols_without_target = [col for col in cat_cols if col != target]

df = pd.get_dummies(df, columns=cat_cols_without_target, drop_first=True)


### Adım 4: Numerik değişkenler için standartlaştırma yapınız.

ss = StandardScaler()

for col in num_cols:
    df[col] = ss.fit_transform(df[[col]])



### Adım 5: Model oluşturunuz.

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import warnings
warnings.simplefilter(action="ignore")

y = df[["Outcome"]]
X = df.drop(["Outcome"], axis=1)


### Model Seçimi

def evaluate_all_models(X, y):
    """
    X: özellik matrisi
    y: hedef değişken
    """
    classifiers = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        SVC(),
        KNeighborsClassifier(),
        GaussianNB()
    ]

    for classifier in classifiers:
        scores = cross_val_score(classifier, X, y, cv=5)
        print(f"{type(classifier).__name__} accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


evaluate_all_models(X, y)

# LogisticRegression accuracy: 0.77 (+/- 0.06)  ****
# DecisionTreeClassifier accuracy: 0.70 (+/- 0.12)
# RandomForestClassifier accuracy: 0.76 (+/- 0.08)
# SVC accuracy: 0.75 (+/- 0.04)
# KNeighborsClassifier accuracy: 0.74 (+/- 0.08)



### Model Tuning  LogisticRegression

model = LogisticRegression()

# Ayarlanacak hiperparametreler
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000, 5000]
}

# GridSearchCV nesnesini oluştur
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

# Modeli eğit ve en iyi hiperparametreleri seç
grid_search.fit(X, y)

# En iyi hiperparametreleri ve accuracy sonucunu yazdır
print("En iyi hiperparametreler: ", grid_search.best_params_)
print("En iyi accuracy sonucu: ", grid_search.best_score_)

# En iyi accuracy sonucu:  0.7734997029114675


### Model Tuning  RandomForestClassifier

model = RandomForestClassifier()

# Ayarlanacak hiperparametreler
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV nesnesini oluştur
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

# Modeli eğit ve en iyi hiperparametreleri seç
grid_search.fit(X, y)

# En iyi hiperparametreleri ve accuracy sonucunu yazdır
print("En iyi hiperparametreler: ", grid_search.best_params_)
print("En iyi accuracy sonucu: ", grid_search.best_score_)

# En iyi accuracy sonucu:  0.7787029963500551