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

df["AdditionalServices"] = df["InternetService"] + df["OnlineSecurity"] + df["OnlineBackup"] + df["DeviceProtection"] + df["TechSupport"]

# Müşterinin fatura ödeme planı hakkında bilgi verir.

df["PayingBills"] = df["Contract"] + df["PaperlessBilling"] + df["PaymentMethod"]

# Müşterinin aile durumu hakkında bilgi verir.

df["PartnerDependents"] = df["Partner"] + df["Dependents"]


# PaymentMethod değişkeni üzerinden Ödeme otomatik mi değilmi üzerinden yeni bir kategorik değişken oluşturuyoruz.

df.loc[(df["PaymentMethod"] == "Credit card (automatic)") | (df["PaymentMethod"] == "Bank transfer (automatic)"), "AutomaticPayment"] = "Yes"
df.loc[~((df["PaymentMethod"] == "Credit card (automatic)") | (df["PaymentMethod"] == "Bank transfer (automatic)")), "AutomaticPayment"] = "No"

