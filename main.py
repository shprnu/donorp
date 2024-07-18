import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

matplotlib.use('TkAgg')

# 0. ЗАГРУЗКА ДАННЫХ
# users_anon_data.csv был разделен, так как github не загружает файлы больше 100Мб
# далее таблицы будут объеденены
users_anon_data = pd.read_csv('./datasets/users_anon_data.csv')
users_anon_data2 = pd.read_csv('./datasets/users_anon_data2.csv')

donations_plan = pd.read_csv('./datasets/anon_donations_plan.csv')
donations_anon = pd.read_csv('./datasets/donations_anon.csv')
clients_anon = pd.read_csv('./datasets/clients_anon.csv')
fundraisings_anon = pd.read_csv('./datasets/fundraisings_anon.csv')

# Объединение users_anon_data и users_anon_data2 по общим столбцам
users_anon_data_all = pd.concat([users_anon_data, users_anon_data2], ignore_index=True)


# 1. ПРЕДОБРАБОТКА ДАННЫХ
# Проверка на наличие пропусков и дубликатов
def data_summary(df):
    return {
        "shape": df.shape,
        "missing_val": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum()
    }


# Получение сводной информации по каждому датасету
summary = {
    "users_anon_data": data_summary(users_anon_data_all),
    "donations_plan": data_summary(donations_plan),
    "donations_anon": data_summary(donations_anon),
    "clients_anon": data_summary(clients_anon),
    "fundraisings_anon": data_summary(fundraisings_anon)
}

print("Общая информация по датасетам:")
for key in summary:
    print('Датасет [%s]' % key)
    print('- размер (%s, %s)' % (summary[key]['shape'][0], summary[key]['shape'][1]))
    print('- пропуски %s' % summary[key]['missing_val'])
    print('- дупликаты %s' % summary[key]['duplicates'])

# Удаление дубликатов, присутствуют только в таблице donations_anon
donations_anon = donations_anon.drop_duplicates()


# Анализ пропусков, присутствуют только в таблице fundraisings_anon
# Пропуски в описании и цели сбора, предсказать непредставляется возможным,
# заполняем числовые поля 0, текстовые - пустыми строками
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna("", inplace=True)
        else:
            df[column].fillna(0, inplace=True)
    return df


fundraisings_anon = fill_missing_values(fundraisings_anon)


# 2. ИСЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ
# Объединение данных пользователей и данных о донациях по идентификатору пользователя
donors_data = pd.merge(users_anon_data_all, donations_anon, left_on='ID', right_on='ID пользователя', how='inner')

# АНАЛИЗ ДОНОРОВ
# Анализ распределения по полу
plt.figure(figsize=(10, 6))
sns.countplot(x='Пол', data=donors_data)
plt.title('Распределение доноров по полу')
plt.show()


# Анализ распределения по возрасту
def get_age(x):
    if x == 'Не указано':
        return None
    else:
        age = datetime.date.today().year - datetime.date(int(x[6:10]), int(x[3:5]), int(x[0:2])).year
        # убираем аномальный возраст
        age = age if (18 <= age <= 65) else None
        return age


donors_data['Возраст'] = donors_data['Дата рождения'].apply(lambda x: get_age(x))

plt.figure(figsize=(10, 6))
sns.histplot(donors_data['Возраст'].dropna(), bins=10)
plt.title('Распределение доноров по возрасту')
plt.show()


# Заполняем регион городами административного значения
def fill_region(rw):
    return rw['Город'] if rw['Регион_y'] == 'Не указан' else rw['Регион_y']


donors_data['Регион'] = donors_data.apply(lambda rw: fill_region(rw), axis=1)

# Анализ распределения по регионам
plt.figure(figsize=(12, 8))
ss = donors_data['Регион'].value_counts().sort_values()
top_regions = donors_data['Регион'].value_counts().head(10).index
sns.countplot(y='Регион', data=donors_data[donors_data['Регион'].isin(top_regions)],
              order=donors_data['Регион'].value_counts(ascending=False).head(10).index)
plt.title('Топ 10 регионов по количеству доноров')
plt.show()

# АНАЛИЗ ПОЖЕРТВОВАНИЙ

