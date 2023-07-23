# TUGAS2_MACHINE_LEARNING_NIM-200401072103_NAMA_HENDRO_GUNAWAN_KELAS_IT-602
Membahas tentang aturan asosiasi (Algoritma Apriori) pada mata kuliah Machine Learning 

Aturan asosiasi (association rules) sering disebut sebagai analisis afinitas (affinity analysis) atau analisis pertalian. Aturan asosiasi merupakan studi mengenai ‘apa bersama apa’ atau “sesuatu memiliki pertalian dengan sesuatu”. Misalnya saat seseorang belanja di supermarket, jika seseorang membeli susu bayi biasanya seseorang juga membeli diapers, dapat dikatakan susu bayi bersama diapers atau susu bayi memiliki pertalian dengan diapers. Karena studi ini diawali pada database transaksi pelanggan, maka studi ini juga disebut “market basket analysis”.
Algoritma apriori adalah sebuah algoritma klasik pada data mining. Algoritma ini menggunakan frekuen itemset untuk menghasilkan aturan asosiasi. Hal ini berdasarkan konsep bahwa subset dari frekuen itemset. Lalu, apa itu frekuen item set?
Frekuen item set merupakan nilai item set minimum yang muncul di himpunan seluruh transaksi (transaksi 1, transaksi 2, transaksi 3, dan seterusnya) atau disebut juga minimum support. Selain frekuen item set, terdapat beberapa terminologi atau istilah lain yang perlu kita pahami pada algoritma apriori seperti:
	K-itemset merupakan item set yang terdiri dari K buah item yang terdapat di dalam himpunan. Misalnya {roti, susu, kopi} merupakan 3-itemset; {madu, es krim} merupakan 2-item set; {keju} merupakan 1-item set. 
	Support merupakan sebuah item set yang memiliki support. Misalnya 10%, jika 10% dari dalam database berisi item-item tersebut.
	Minimum support merupakan nilai minimum yang telah ditentukan sebelum algoritma apriori dimulai.
