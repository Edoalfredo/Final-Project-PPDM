from lib import*

model = joblib.load('multinomial_nb_10 percent_model.pkl')

def predict_text(input_text):
    reviews_data = pd.read_excel("reviews_Preprocessing.xlsx", usecols=["Label", "reviews_tokens_stemmed"])
    reviews_data.columns = ["label", "reviews"]
    def join_text_list(texts):
        texts = ast.literal_eval(texts)
        return ' '.join([text for text in texts])
    reviews_data["reviews_join"] = reviews_data["reviews"].apply(join_text_list)

    label = reviews_data["label"]
    text = reviews_data["reviews_join"]

    train_data, test_data, train_labels, test_labels = train_test_split(text, label, test_size=0.2, random_state=42)

    positive_count = (train_labels == 1).sum()
    negative_count = (train_labels == 0).sum()
    total_count = len(train_labels)
    positive_ratio = positive_count / total_count
    negative_ratio = negative_count / total_count

    cvect = CountVectorizer()
    TF_vector_train = cvect.fit_transform(train_data)

    # Normalisasi TF vector pada train set
    normalized_TF_vector_train = normalize(TF_vector_train, norm='l1', axis=1)

    # Perhitungan TF vector pada test set menggunakan CountVectorizer yang sudah dilatih pada train set
    TF_vector_test = cvect.transform(test_data)

    # Normalisasi TF vector pada test set
    normalized_TF_vector_test = normalize(TF_vector_test, norm='l1', axis=1)

    # Persentase fitur yang ingin dipilih setelah seleksi (10%)
    percent = 10

    # Menghitung jumlah fitur yang diinginkan berdasarkan persentase
    k = int(percent / 100 * normalized_TF_vector_train.shape[1])

    # Menerapkan seleksi fitur dengan chi-square pada train set
    selector = SelectPercentile(chi2, percentile=percent)
    tf_mat_train_selected = selector.fit_transform(normalized_TF_vector_train, train_labels)

    # Mengaplikasikan seleksi fitur yang sama pada test set
    tf_mat_test_selected = selector.transform(normalized_TF_vector_test)

    input_vector = cvect.transform([input_text])

    # Normalisasi vektor fitur
    normalized_input_vector = normalize(input_vector, norm='l1', axis=1)

    # Terapkan seleksi fitur pada vektor fitur input
    input_vector_selected = selector.transform(normalized_input_vector)

    # Lakukan prediksi menggunakan model yang telah Anda bangun
    prediction = model.predict(input_vector_selected)

    # Cetak output klasifikasi
    if prediction == 0:
        st.write('Ulasan negatif:angry:')
        st.image('ANGRY.png')
    else:
        st.write('Ulasan positif:smiley:')
        st.image('HAPPY.png')