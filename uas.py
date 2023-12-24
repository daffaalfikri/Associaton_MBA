import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv("Groceries data.csv")

st.title("Market Analyst MBA")

def get_data(itemDescription='', year=''):
    data = df.copy()
    filtered = data.loc[
        (data["itemDescription"].str.contains(itemDescription)) &
        (data["year"].astype(str).str.contains(year))
    ]
    return filtered if not filtered.empty else "No Result"


def user_input_features():
    Product = st.selectbox("Member_number", ['1808', '2552', '2300', '1187',
                           '3037', '4941','4501'])
    itemDescription = st.selectbox("itemDescription", ['tropical fruit', 'whole milk', 'pip fruit',
                                   'other vegetables', 'whole milk', 'rolls/buns', 'other vegetables', 'pot plants', 'whole milk'])
    year = st.select_slider("Bulan", list(map(str, range(1, 12))))
    return itemDescription, year, Product


itemDescription, year, Product = user_input_features()

data = get_data(itemDescription.lower(), year)

# Fungsi untuk mendapatkan frequent itemsets dan association rules
def get_apriori_results(transactions, min_support, min_confidence):
    # Membuat format data yang sesuai untuk algoritma Apriori
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Menerapkan algoritma Apriori untuk mendapatkan itemset yang sering muncul
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # Menerapkan aturan asosiasi dari itemset yang sering muncul
    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return frequent_itemsets, rules

# Menampilkan tampilan aplikasi Streamlit


def main():
    st.title("Apriori Algoritma Association")

    # Contoh data transaksi
    transactions = [
        ['sausage', 'whole milk', 'curd'],
        ['sausage', 'curd'],
        ['whole milk', 'sausage', 'curd'],
        ['curd', 'whole milk'],
        ['sausage']
    ]

    # Parameter untuk algoritma Apriori
    min_support = st.slider("Minimal Support", 0.0, 0.8, 0.2)
    min_confidence = st.slider("Minimal Confidence", 0.0, 1.0, 0.7)
    
    # Mendapatkan hasil dari algoritma Apriori
    frequent_itemsets, rules = get_apriori_results(
        transactions, min_support, min_confidence)

    # Menampilkan hasil
    st.subheader("Frequent Itemsets:")
    st.write(frequent_itemsets)

    st.subheader("Association Rules:")
    st.write(rules)


if __name__ == "__main__":
    main()
