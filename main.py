import pandas as pd

# ЗАГРУЗКА ДАННЫХ
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

