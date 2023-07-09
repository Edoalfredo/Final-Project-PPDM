from lib import*

model = joblib.load('multinomial_nb_10 percent_model.pkl')

def predict_text(input_text):
    reviews_data = pd.read_excel("reviews_Preprocessing.xlsx", usecols=["Label", "Stemming"])
    reviews_data.columns = ["label", "reviews"]
    def join_text_list(texts):
        texts = ast.literal_eval(texts)
        return ' '.join([text for text in texts])
    reviews_data["reviews"] = reviews_data["reviews"].apply(join_text_list)

    label = reviews_data["label"]
    text = reviews_data["reviews"]

    train_data, test_data, train_labels, test_labels = train_test_split(text, label, test_size=0.2, random_state=42)

    positive_count = (train_labels == 1).sum()
    negative_count = (train_labels == 0).sum()
    total_count = len(train_labels)

    cvect = CountVectorizer()
    TF_vector_train = cvect.fit_transform(train_data)

    # Perhitungan TF vector pada test set menggunakan CountVectorizer yang sudah dilatih pada train set
    TF_vector_test = cvect.transform(test_data)

    # Persentase fitur yang ingin dipilih setelah seleksi (10%)
    percent = 10

    # Menghitung jumlah fitur yang diinginkan berdasarkan persentase
    k = int(percent / 100 * TF_vector_train.shape[1])

    # Menerapkan seleksi fitur dengan chi-square pada train set
    selector = SelectPercentile(chi2, percentile=percent)
    tf_mat_train_selected = selector.fit_transform(TF_vector_train, train_labels)

    # Mengaplikasikan seleksi fitur yang sama pada test set
    tf_mat_test_selected = selector.transform(TF_vector_test)

    input_vector = cvect.transform([input_text])

    # Terapkan seleksi fitur pada vektor fitur input
    input_vector_selected = selector.transform(input_vector)

    # Lakukan prediksi menggunakan model yang telah Anda bangun
    prediction = model.predict(input_vector_selected)

    # Cetak output klasifikasi
    if prediction == 0:
        st.write('Ulasan negatif:angry:')
        st.image('ANGRY.png')
    else:
        st.write('Ulasan positif:smiley:')
        st.image('HAPPY.png')
    