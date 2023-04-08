import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


### İş Problemi

'''

Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi
ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

'''

### Veri seti & Değişkenler

'''

Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu
gösterir.


Telco-Customer-Churn.csv

CustomerId:  Müşteri İd’si
Gender:  Cinsiyet
SeniorCitizen:  Müşterinin yaşlı olup olmadığı (1, 0)
Partner:  Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
Dependents:  Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
tenure:  Müşterinin şirkette kaldığı ay sayısı
PhoneService:  Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
MultipleLines:  Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
InternetService:  Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
OnlineSecurity:  Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
OnlineBackup:  Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
DeviceProtection:  Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
TechSupport:  Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingTV:  Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
StreamingMovies:  Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
Contract:  Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
PaperlessBilling:  Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
PaymentMethod:  Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
MonthlyCharges:  Müşteriden aylık olarak tahsil edilen tutar
TotalCharges:  Müşteriden tahsil edilen toplam tutar
Churn:  Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

'''

#######################
## Görev 1: Keşifçi Veri Analizi
#######################

df = pd.read_csv("Miuul_Course_1/Feature-Engineering/Dataset/Telco-Customer-Churn.csv")
df.head()
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

### Adım 2: Numerik ve kategorik değişkenleri yakalayınız.

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

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
num_cols.append("TotalCharges")


### Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.

#### Kategorik değişkenlerin analizi

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

#### Numerik değişkenlerin analizi

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col)


### Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)

target = "Churn"

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, target, col)

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


### Adım 6: Eksik gözlem analizi yapınız

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


### Adım 7: Korelasyon analizi yapınız.

sns.heatmap(df.corr())


#######################
## Görev 2: Feature Engineering
#######################

### Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

#### Eksik değerler

missing_values_table(df) # sadece bir değişkende ve 11 tane olduğu için siliyoruz

df.dropna(inplace=True)

#### Aykırı gözlemler

for col in num_cols:
    print(check_outlier(df, col)) # Aykırı değer yok!!

### Adım 2: Yeni değişkenler oluşturunuz.

# Müşterinin ne kadar yüzde ödediğini gösterir.
df["RateCharges"] = df["MonthlyCharges"] / df["TotalCharges"]


# Müşterinin cinsiyeti ve yaşlı olup olmadığı.
# df.loc[(df["SeniorCitizen"] == 0) & (df["gender"] == "Male")]

df.loc[(df["SeniorCitizen"] == 0) & (df["gender"] == "Female"), "GenderCitizenCat"] = "YoungFemale"
df.loc[(df["SeniorCitizen"] == 1) & (df["gender"] == "Female"), "GenderCitizenCat"] = "OldFemale"
df.loc[(df["SeniorCitizen"] == 0) & (df["gender"] == "Male"), "GenderCitizenCat"] = "YoungMale"
df.loc[(df["SeniorCitizen"] == 1) & (df["gender"] == "Male"), "GenderCitizenCat"] = "OldMale"


# Müşterinin internet hizmetleri ile ilgili olarak herhangi bir ek hizmete sahip olup olmadığını gösterir.

df["AdditionalServices"] = df["InternetService"] + "_" + df["OnlineSecurity"] + "_" + df["OnlineBackup"] + "_" + df["DeviceProtection"] + "_" + df["TechSupport"]

# Müşterinin fatura ödeme planı hakkında bilgi verir.

df["PayingBills"] = df["Contract"] + "_" + df["PaperlessBilling"] + "_" + df["PaymentMethod"]

# Müşterinin aile durumu hakkında bilgi verir.

df["PartnerDependents"] = df["Partner"] + "_" + df["Dependents"]


# PaymentMethod değişkeni üzerinden Ödeme otomatik mi değilmi üzerinden yeni bir kategorik değişken oluşturuyoruz.

df.loc[(df["PaymentMethod"] == "Credit card (automatic)") | (df["PaymentMethod"] == "Bank transfer (automatic)"), "AutomaticPayment"] = "Yes"
df.loc[~((df["PaymentMethod"] == "Credit card (automatic)") | (df["PaymentMethod"] == "Bank transfer (automatic)")), "AutomaticPayment"] = "No"

# Tenure Değişkenine göre müşterilerin yeni, orta, eski vb gibi categoriler oluşturacağız.

df["tenure"].describe().T

df.loc[df["tenure"] < 12, "TenureCat"] = "Yeni"
df.loc[(df["tenure"] > 12) & (df["tenure"] < 24), "TenureCat"] = "Orta"
df.loc[(df["tenure"] > 24) & (df["tenure"] < 36), "TenureCat"] = "Eski"
df.loc[(df["tenure"] > 36), "TenureCat"] = "Cok_Eski"

df["TenureCat"].value_counts()


# Müşterinin ne kadar ek hizmet aldığı


df["NumOfExtraServices"] = df["OnlineSecurity"] + "_" + df["OnlineBackup"] + "_" + df["DeviceProtection"] + "_" + df["TechSupport"] + "_" + df["StreamingTV"] + "_" + df["StreamingMovies"]

df["NumOfExtraServices"] = [row.split("_") for row in df["NumOfExtraServices"]]
df["NumOfExtraServices"] = [row.count("Yes") for row in df["NumOfExtraServices"]]


# No phone service ve No internet service'i No ya dönüştürdük çünkü farklı bir değişkende Phone sevice ve Internet sevice alıp almadığını belirtiyorduk.

for i in df.columns:
    df[i] = ["No" if ((row == "No phone service") | (row == "No internet service")) else row for row in df[i]]




### Adım 3: Encoding işlemlerini gerçekleştiriniz.


df_encoding = df.copy()

#### Encoding sadece yes-no olan değişkenleri encoding yapıyoruz get dummies yapınca fazla değişken oluşmaması için

# list(df["Dependents"].unique()) # ["No", "Yes"]
# list(df["Dependents"].unique()) # ["Yes", "No"]

yes_no_cols = [col for col in df_encoding.columns if (list(df_encoding[col].unique()) == ["Yes", "No"]) | (list(df_encoding[col].unique()) == ["No", "Yes"])]

for i in yes_no_cols:
    df_encoding[i] = [1 if row == "Yes" else 0 for row in df_encoding[i]]

df_encoding[yes_no_cols] = df_encoding[yes_no_cols].astype("uint8")

#### Encoding with get_dummies

cols_for_get_dummies = ["gender", "InternetService", "Contract", "PaymentMethod", "GenderCitizenCat", "AdditionalServices", "PayingBills", "PartnerDependents", "TenureCat", "NumOfExtraServices"]


df_encoding = pd.get_dummies(df_encoding, columns=cols_for_get_dummies, drop_first=True)

df_encoding.info()



### Adım 4: Numerik değişkenler için standartlaştırma yapınız.

df_standard = df_encoding.copy()

ss = StandardScaler()

for col in num_cols:
    df_standard[[col]] = ss.fit_transform(df_standard[[col]])

df_standard[num_cols]


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

final_df = df_standard.copy()

y = final_df[["Churn"]]
X = final_df.drop(["Churn", "customerID"], axis=1)


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


# LogisticRegression accuracy: 0.80 (+/- 0.01)
# DecisionTreeClassifier accuracy: 0.72 (+/- 0.02)
# RandomForestClassifier accuracy: 0.79 (+/- 0.02)
# SVC accuracy: 0.80 (+/- 0.02)
# KNeighborsClassifier accuracy: 0.78 (+/- 0.02)
# GaussianNB accuracy: 0.67 (+/- 0.01)


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

# En iyi accuracy sonucu:  0.8060289893754152



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

# En iyi accuracy sonucu:  0.8017627772537435




### Model Tuning  SVC

model = SVC()


# GridSearchCV için parametrelerin değerlerini belirleyelim
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.1, 0.01]
}


# GridSearchCV ile en iyi parametreleri bulalım
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

# En iyi parametreleri ve test seti skorunu yazdıralım
print("En iyi parametreler: ", grid_search.best_params_)
print("Test seti skoru: ", grid_search.best_score_)

# Test seti skoru:  0.8017627772537435
