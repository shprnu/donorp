import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np

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
top_regions = donors_data['Регион'].value_counts().head(10).index
sns.countplot(y='Регион', data=donors_data[donors_data['Регион'].isin(top_regions)],
              order=donors_data['Регион'].value_counts(ascending=False).head(10).index)
plt.title('Топ 10 регионов по количеству доноров')
plt.show()

# АНАЛИЗ ПОЖЕРТВОВАНИЙ
# Объединение данных пользователей и данных о пожертвованиях по идентификатору пользователя
clients_anon['ID пользователя'] = pd.to_numeric(clients_anon['ID пользователя'], errors='coerce')

donors_clients_data = pd.merge(clients_anon, fundraisings_anon, left_on='ID пользователя',
                               right_on='ID пользователя', how='inner')


# Анализ распределения по возрасту
def get_age_2(x):
    if x == 'Не указано':
        return None
    else:
        age = datetime.date.today().year - datetime.date(int(x[0:4]), int(x[5:7]), int(x[8:10])).year
        # убираем аномальный возраст
        age = age if (18 <= age <= 99) else None
        return age


donors_clients_data['Возраст'] = donors_clients_data['Дата рождения'].apply(lambda x: get_age_2(x))

plt.figure(figsize=(10, 6))
sns.histplot(donors_clients_data['Возраст'].dropna(), bins=5)
plt.title('Распределение жертвователей по возрасту')
plt.show()

# Анализ распределения по суммам
top_clients = donors_clients_data.groupby('ID пользователя')['Собрано'].sum().sort_values(ascending=False)
palette_color = sns.color_palette('bright')
l_data = top_clients.tolist()[0:5]
l_data.append(sum(top_clients.tolist()[5:]))
l_label = list(map(str, list(map(int, top_clients.index.values.tolist()[0:5]))))
l_label.append('Other')
plt.pie(l_data, labels=l_label, colors=palette_color, autopct='%.0f%%')
plt.title('Распределение жертвователей по сумме пожертвований')
plt.show()


# МЕТРИКИ РЕЗУЛЬТАТОВ ДЕЯТЕЛЬНОСТИ DONORSEARCH
# Преобразование дат в datetime
def convert_dt(x):
    return datetime.datetime.strptime(x, '%d.%m.%Y').strftime('%m.%d.%Y')


donations_anon['Дата донации'] = donations_anon['Дата донации'].apply(lambda x: convert_dt(x))
donations_anon['Дата донации'] = pd.to_datetime(donations_anon['Дата донации'])
# Убираем аномальные даты
donations_anon = donations_anon[donations_anon['Дата донации'] <= pd.Timestamp.today()]

fundraisings_anon['Дата начала сбора'] = fundraisings_anon['Дата начала сбора'].apply(lambda x: convert_dt(x))
fundraisings_anon['Дата начала сбора'] = pd.to_datetime(fundraisings_anon['Дата начала сбора'])
# Убираем аномальные даты
fundraisings_anon = fundraisings_anon[fundraisings_anon['Дата начала сбора'] <= pd.Timestamp.today()]

# DAU и MAU
# Расчет DAU
donations_anon['date'] = donations_anon['Дата донации'].dt.date
fundraisings_anon['date'] = fundraisings_anon['Дата начала сбора'].dt.date

dau_donations = donations_anon.groupby('date').size().reset_index(name='DAU')
dau_fundraisings = fundraisings_anon.groupby('date').size().reset_index(name='DAU')

# Расчет MAU
donations_anon['month'] = donations_anon['Дата донации'].dt.to_period('M')
fundraisings_anon['month'] = fundraisings_anon['Дата начала сбора'].dt.to_period('M')

mau_donations = donations_anon.groupby('month').size().reset_index(name='MAU')
mau_fundraisings = fundraisings_anon.groupby('month').size().reset_index(name='MAU')

print("DAU Donations", dau_donations)
print("DAU Fundraisings", dau_fundraisings)
print("MAU Donations", mau_donations)
print("MAU Fundraisings", mau_fundraisings)

# Расчет количества донаций на одного пользователя
avg_donation_amount = donations_anon.groupby('ID пользователя')['ID'].count() \
    .reset_index(name='avg_donation_amount')
# Расчет средней суммы пожертвований на одного пользователя
avg_fundraising_amount = fundraisings_anon.groupby('ID пользователя')['Собрано'].mean()\
    .reset_index(name='avg_fundraising_amount')

# Расчет средней продолжительности жизни пользователя на платформе (в днях)
donations_anon['user_lifetime'] = (donations_anon['Дата донации'].max() - donations_anon['Дата донации'].min())
fundraisings_anon['user_lifetime'] = (fundraisings_anon['Дата начала сбора'].max() -
                                      fundraisings_anon['Дата начала сбора'].min())

avg_user_lifetime_donations = donations_anon.groupby('ID пользователя')['user_lifetime'].mean()\
    .reset_index(name='avg_user_lifetime')
avg_user_lifetime_fundraisings = fundraisings_anon.groupby('ID пользователя')['user_lifetime'].mean()\
    .reset_index(name='avg_user_lifetime')

# LTV
# Расчет LTV
ltv_donations = pd.merge(avg_donation_amount, avg_user_lifetime_donations, on='ID пользователя')
ltv_donations['LTV'] = ltv_donations['avg_donation_amount'] * ltv_donations['avg_user_lifetime']

ltv_fundraisings = pd.merge(avg_fundraising_amount, avg_user_lifetime_fundraisings, on='ID пользователя')
ltv_fundraisings['LTV'] = ltv_fundraisings['avg_fundraising_amount'] * ltv_fundraisings['avg_user_lifetime']

print("LTV Donations", ltv_donations)
print("LTV Fundraisings", ltv_fundraisings)

# RETENTION RATE
# Расчет коэффициента удержания пользователей
def calculate_retention_rate(df, date_column):
    df['registration_month'] = df[date_column].dt.to_period('M')
    cohort_sizes = df.groupby('registration_month').size()
    retention_data = df.groupby(['registration_month', df[date_column].dt.to_period('M')]).size().unstack(fill_value=0)
    retention_rate = retention_data.divide(cohort_sizes, axis=0)
    return retention_rate


retention_donations = calculate_retention_rate(donations_anon, 'Дата донации')
retention_fundraisings = calculate_retention_rate(fundraisings_anon, 'Дата начала сбора')

print("Retention Rate Donations", retention_donations)
print("Retention Rate Fundraisings", retention_fundraisings)

# DONATION FREQUENCY И DONATION AMOUNT GROWTH
# Частота донаций (в среднем на одного пользователя)
donation_frequency = donations_anon.groupby('ID пользователя').size().mean()
fundraising_frequency = fundraisings_anon.groupby('ID пользователя').size().mean()

# Рост суммы пожертвований с течением времени
donations_anon['year_month'] = donations_anon['Дата донации'].dt.to_period('M')
fundraisings_anon['year_month'] = fundraisings_anon['Дата начала сбора'].dt.to_period('M')

donation_amount_growth = donations_anon.groupby('year_month')['ID'].count().pct_change().fillna(0)\
    .reset_index(name='Donation Amount Growth')
fundraising_amount_growth = fundraisings_anon.groupby('year_month')['Собрано'].sum().pct_change().fillna(0)\
    .reset_index(name='Fundraising Amount Growth')

print("Donation Amount Growth", donation_amount_growth)
print("Fundraising Amount Growth", fundraising_amount_growth)

# РАСЧЕТ РЕЗУЛЬТАТОВ РАБОТЫ НКО DS
# Определение периода (например, последние 12 месяцев)
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(months=12)

# Фильтрация данных по выбранному периоду
filtered_donations = donations_anon[(donations_anon['Дата донации'] >= start_date) &
                                    (donations_anon['Дата донации'] <= end_date)]
filtered_fundraisings = fundraisings_anon[(fundraisings_anon['Дата начала сбора'] >= start_date) &
                                          (fundraisings_anon['Дата начала сбора'] <= end_date)]

# Количество уникальных доноров
num_donors = filtered_donations['ID пользователя'].nunique()

# Количество сделанных донаций
num_donations = filtered_donations.shape[0]

# Общая сумма собранных пожертвований
total_donation_amount = filtered_fundraisings['Собрано'].sum()

# Сводная таблица результатов
print("Период", f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
print("Количество привлеченных доноров", num_donors)
print("Количество сделанных донаций", num_donations)
print("Общая сумма собранных пожертвований", total_donation_amount)

# Выводы и рекомендации по результатам работы НКО DonorSearch
# ВЫВОДЫ:
# 1. За последний год НКО DonorSearch привлекла значительное количество новых доноров,
# демонстрируя успешную работу по вовлечению новых участников.
# 2. Ежедневная (DAU) и ежемесячная (MAU) активность пользователей показывает стабильное использование платформы,
# что свидетельствует о высокой заинтересованности и вовлеченности пользователей.
# 3. За отчетный период было сделано значительное количество донаций, что подчеркивает активное участие доноров.
# 4. Общая сумма собранных пожертвований является значительной, указывая на успех платформы в сборе средств.
# 5. Доноры неравномерно по полу, с преобладающей долей мужчин.
# 6. Основная возрастная группа активных пользователей составляет от 30 до 40 лет для доноров
# и от 25 до 45 лет для жертвователей, что соответствует активному возрасту.
# 7. Доноры в основном сосредоточены в крупных городах и развитых регионах,
# что может быть связано с лучшей инфраструктурой и доступностью медицинских учреждений.
# 8. Customer Lifetime Value (LTV) указывает на значительную ценность, которую пользователи приносят платформе
# за все время своего взаимодействия.
# 9. Рост суммы пожертвований (Donation Amount Growth) показывает положительную динамику,
# что свидетельствует об успешных кампаниях по сбору средств.
# РЕКОМЕНДАЦИИ:
# 1. Разработайте программы лояльности для доноров, чтобы поощрять повторные донации и участие в кампаниях.
# 2. Регулярно собирайте обратную связь от текущих доноров, чтобы улучшить пользовательский опыт и сделать
# платформу более привлекательной для новых участников.
# 3. Разработайте и распространяйте материалы, объясняющие, как пожертвования используются для достижения целей НКО.
# Это может стимулировать жертвователей к увеличению суммы пожертвований.
#

