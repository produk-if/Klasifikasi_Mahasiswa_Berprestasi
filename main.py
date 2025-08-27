import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FuzzyKNN:
    def __init__(self, k=5, m=2):
        """
        Fuzzy K-Nearest Neighbors Classifier
        
        Parameters:
        k: number of neighbors
        m: fuzzy parameter (typically 2)
        """
        self.k = k
        self.m = m
        self.X_train = None
        self.y_train = None
        self.scaler = StandardScaler()
        
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def fit(self, X_train, y_train):
        """Fit the model with training data"""
        self.X_train = self.scaler.fit_transform(X_train)
        self.y_train = y_train
        self.classes = np.unique(y_train)
        return self
    
    def predict(self, X_test, return_membership=False):
        """Predict classes for test data"""
        X_test_scaled = self.scaler.transform(X_test)
        predictions = []
        membership_values = []
        
        for test_point in X_test_scaled:
            # Calculate distances to all training points
            distances = []
            for train_point in self.X_train:
                dist = self.euclidean_distance(test_point, train_point)
                distances.append(dist)
            
            # Get k nearest neighbors
            distances = np.array(distances)
            k_indices = np.argsort(distances)[:self.k]
            k_distances = distances[k_indices]
            k_labels = self.y_train[k_indices]
            
            # Calculate fuzzy membership for each class
            class_memberships = {}
            for class_label in self.classes:
                numerator = 0
                denominator = 0
                
                for i, (dist, label) in enumerate(zip(k_distances, k_labels)):
                    if dist == 0:  # Handle zero distance
                        if label == class_label:
                            class_memberships[class_label] = 1.0
                        else:
                            class_memberships[class_label] = 0.0
                        break
                    
                    weight = 1 / (dist ** (2 / (self.m - 1)))
                    denominator += weight
                    if label == class_label:
                        numerator += weight
                
                if dist != 0:
                    class_memberships[class_label] = numerator / denominator if denominator > 0 else 0
            
            # Predict class with highest membership
            predicted_class = max(class_memberships.keys(), key=lambda x: class_memberships[x])
            predictions.append(predicted_class)
            membership_values.append(class_memberships)
        
        if return_membership:
            return np.array(predictions), membership_values
        return np.array(predictions)

def load_and_process_data():
    """Load and process both datasets"""
    print("=" * 60)
    print("MEMUAT DAN MEMPROSES DATA")
    print("=" * 60)
    
    try:
        # Load datasets
        mahasiswa_df = pd.read_csv('mahasiswa_data_20250826_152655.csv')
        prestasi_df = pd.read_csv('prestasi.csv')
        
        print(f"Data mahasiswa: {len(mahasiswa_df)} records")
        print(f"Data prestasi: {len(prestasi_df)} records")
        
        # Extract academic features from mahasiswa data
        academic_features = []
        
        for idx, row in mahasiswa_df.iterrows():
            nim = row['nim']
            
            # Extract semester data
            semester_data = []
            for sem in range(1, 16):  # semester 1-15
                ips_col = f'khs{sem}_ips'
                ipk_col = f'khs{sem}_ipk'
                sks_col = f'khs{sem}_sksTotal'
                
                if pd.notna(row.get(ips_col)):
                    semester_data.append({
                        'ips': row[ips_col],
                        'ipk': row[ipk_col],
                        'sks': row.get(sks_col, 0)
                    })
            
            if semester_data:
                # Calculate academic metrics
                final_ipk = semester_data[-1]['ipk']
                final_sks = semester_data[-1]['sks']
                ips_values = [s['ips'] for s in semester_data]
                
                avg_ips = np.mean(ips_values)
                ips_std = np.std(ips_values) if len(ips_values) > 1 else 0
                stability_score = 1 / (1 + ips_std)  # Higher = more stable
                semester_count = len(semester_data)
                
                # Academic trend
                if len(ips_values) > 1:
                    trend = 1 if ips_values[-1] - ips_values[0] > 0.1 else (
                        -1 if ips_values[-1] - ips_values[0] < -0.1 else 0)
                else:
                    trend = 0
                
                academic_features.append({
                    'nim': nim,
                    'final_ipk': final_ipk,
                    'final_sks': final_sks,
                    'avg_ips': avg_ips,
                    'stability_score': stability_score,
                    'semester_count': semester_count,
                    'academic_trend': trend,
                    'gender': 1 if row.get('jenisKelamin') == 'L' else 0,
                    'graduation_status': 1 if row.get('lulus') == 'True' else 0
                })
        
        academic_df = pd.DataFrame(academic_features)
        
        # Process prestasi data
        prestasi_summary = prestasi_df.groupby('id_mahasiswa').agg({
            'tingkat': lambda x: len(x),  # total prestasi count
            'kategori': lambda x: (x == 'akademik').sum(),  # academic achievement count
            'jenis_prestasi': lambda x: (x == 'individu').sum()  # individual achievement count
        }).reset_index()
        
        prestasi_summary.columns = ['nim', 'total_prestasi', 'prestasi_akademik', 'prestasi_individu']
        prestasi_summary['prestasi_non_akademik'] = prestasi_summary['total_prestasi'] - prestasi_summary['prestasi_akademik']
        
        # Calculate achievement levels
        tingkat_scores = prestasi_df.groupby('id_mahasiswa')['tingkat'].apply(
            lambda x: sum([3 if t == 'internasional' else 2 if t == 'nasional' else 1 for t in x])
        ).reset_index()
        tingkat_scores.columns = ['nim', 'achievement_level_score']
        
        prestasi_summary = prestasi_summary.merge(tingkat_scores, on='nim', how='left')
        prestasi_summary['achievement_level_score'] = prestasi_summary['achievement_level_score'].fillna(0)
        
        # Ensure nim columns have the same data type for merging
        academic_df['nim'] = academic_df['nim'].astype(str)
        prestasi_summary['nim'] = prestasi_summary['nim'].astype(str)
        
        print(f"Mahasiswa dengan data akademik: {len(academic_df)}")
        print(f"Mahasiswa dengan prestasi: {len(prestasi_summary)}")
        
        return academic_df, prestasi_summary
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def create_combined_dataset(academic_df, prestasi_summary):
    """Combine academic and achievement data"""
    print("\n" + "=" * 60)
    print("MENGGABUNGKAN DAN MEMBUAT FITUR")
    print("=" * 60)
    
    # Merge datasets
    combined_df = academic_df.merge(prestasi_summary, on='nim', how='left')
    
    # Fill missing values for students without achievements
    achievement_cols = ['total_prestasi', 'prestasi_akademik', 'prestasi_non_akademik', 
                       'prestasi_individu', 'achievement_level_score']
    for col in achievement_cols:
        combined_df[col] = combined_df[col].fillna(0)
    
    # Create composite scores
    combined_df['academic_score'] = (
        combined_df['final_ipk'] * 0.4 + 
        combined_df['stability_score'] * 0.3 + 
        (combined_df['academic_trend'] + 1) * 0.3  # normalize trend to 0-2, then weight
    )
    
    combined_df['achievement_score'] = (
        combined_df['total_prestasi'] * 0.3 +
        combined_df['achievement_level_score'] * 0.4 +
        combined_df['prestasi_akademik'] * 0.3
    )
    
    # Create labels based on comprehensive criteria
    def create_label(row):
        # Multi-criteria approach for labeling
        academic_threshold = row['final_ipk'] >= 3.50 and row['stability_score'] >= 0.5
        achievement_threshold = row['total_prestasi'] >= 2 or row['achievement_level_score'] >= 3
        
        # Berprestasi if meets academic AND achievement criteria
        if academic_threshold and achievement_threshold:
            return 1
        # Also berprestasi if exceptional in one area
        elif row['final_ipk'] >= 3.80 or row['total_prestasi'] >= 4:
            return 1
        else:
            return 0
    
    combined_df['label'] = combined_df.apply(create_label, axis=1)
    
    print(f"Total mahasiswa dalam dataset gabungan: {len(combined_df)}")
    print(f"Mahasiswa berprestasi: {sum(combined_df['label'])} ({sum(combined_df['label'])/len(combined_df)*100:.1f}%)")
    print(f"Mahasiswa tidak berprestasi: {len(combined_df) - sum(combined_df['label'])} ({(len(combined_df) - sum(combined_df['label']))/len(combined_df)*100:.1f}%)")
    
    return combined_df

def visualize_data(combined_df):
    """Create visualizations of the dataset"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analisis Dataset Mahasiswa Berprestasi', fontsize=16, fontweight='bold')
    
    # Distribution of labels
    label_counts = combined_df['label'].value_counts()
    axes[0,0].pie(label_counts.values, labels=['Tidak Berprestasi', 'Berprestasi'], 
                  autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    axes[0,0].set_title('Distribusi Kelas')
    
    # IPK distribution by class
    axes[0,1].hist([combined_df[combined_df['label']==0]['final_ipk'], 
                    combined_df[combined_df['label']==1]['final_ipk']], 
                   bins=15, alpha=0.7, label=['Tidak Berprestasi', 'Berprestasi'], 
                   color=['#ff9999', '#66b3ff'])
    axes[0,1].set_xlabel('IPK Akhir')
    axes[0,1].set_ylabel('Frekuensi')
    axes[0,1].set_title('Distribusi IPK berdasarkan Kelas')
    axes[0,1].legend()
    
    # Achievement count by class
    axes[1,0].hist([combined_df[combined_df['label']==0]['total_prestasi'], 
                    combined_df[combined_df['label']==1]['total_prestasi']], 
                   bins=10, alpha=0.7, label=['Tidak Berprestasi', 'Berprestasi'], 
                   color=['#ff9999', '#66b3ff'])
    axes[1,0].set_xlabel('Jumlah Prestasi')
    axes[1,0].set_ylabel('Frekuensi')
    axes[1,0].set_title('Distribusi Jumlah Prestasi berdasarkan Kelas')
    axes[1,0].legend()
    
    # Scatter plot: IPK vs Achievement Score
    scatter = axes[1,1].scatter(combined_df['final_ipk'], combined_df['achievement_score'], 
                               c=combined_df['label'], alpha=0.6, cmap='coolwarm')
    axes[1,1].set_xlabel('IPK Akhir')
    axes[1,1].set_ylabel('Skor Prestasi')
    axes[1,1].set_title('IPK vs Skor Prestasi')
    plt.colorbar(scatter, ax=axes[1,1], label='Kelas (0=Tidak, 1=Berprestasi)')
    
    plt.tight_layout()
    plt.show()

def run_fuzzy_knn_experiment(combined_df):
    """Run Fuzzy K-NN classification experiment"""
    print("\n" + "=" * 60)
    print("EKSPERIMEN FUZZY K-NN")
    print("=" * 60)
    
    # Prepare features and target
    feature_columns = ['final_ipk', 'avg_ips', 'stability_score', 'semester_count', 
                      'academic_trend', 'total_prestasi', 'prestasi_akademik', 
                      'prestasi_non_akademik', 'achievement_level_score', 
                      'academic_score', 'achievement_score']
    
    X = combined_df[feature_columns]
    y = combined_df['label'].values
    
    print("Fitur yang digunakan:")
    for i, feature in enumerate(feature_columns, 1):
        print(f"{i:2d}. {feature}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"\nData training: {len(X_train)} samples")
    print(f"Data testing: {len(X_test)} samples")
    
    # Test different K values
    k_values = [3, 5, 7, 9, 11]
    results = {}
    
    for k in k_values:
        print(f"\nMenguji dengan K = {k}:")
        
        # Train Fuzzy K-NN
        fknn = FuzzyKNN(k=k, m=2)
        fknn.fit(X_train, y_train)
        
        # Predict
        y_pred, membership_values = fknn.predict(X_test, return_membership=True)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[k] = accuracy
        
        print(f"  Akurasi: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show detailed results for best K
        if k == 5:  # Show details for K=5
            print("\n  Contoh nilai keanggotaan fuzzy (5 sampel pertama):")
            for i in range(min(5, len(membership_values))):
                memberships = membership_values[i]
                print(f"    Sampel {i+1}: Tidak Berprestasi={memberships[0]:.3f}, Berprestasi={memberships[1]:.3f}")
                print(f"    Prediksi: {'Berprestasi' if y_pred[i] == 1 else 'Tidak Berprestasi'}, Aktual: {'Berprestasi' if y_test[i] == 1 else 'Tidak Berprestasi'}")
            
            print(f"\n  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Tidak Berprestasi', 'Berprestasi']))
            
            print(f"\n  Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    k_list = list(results.keys())
    acc_list = list(results.values())
    
    plt.plot(k_list, acc_list, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Nilai K')
    plt.ylabel('Akurasi')
    plt.title('Performa Fuzzy K-NN dengan Berbagai Nilai K')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_list)
    
    for i, (k, acc) in enumerate(zip(k_list, acc_list)):
        plt.annotate(f'{acc:.3f}', (k, acc), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    best_k = max(results.keys(), key=lambda k: results[k])
    print(f"\nNilai K terbaik: {best_k} dengan akurasi: {results[best_k]:.4f}")
    
    return results

def main():
    """Main function to run the complete analysis"""
    print("🎓 SISTEM KLASIFIKASI MAHASISWA BERPRESTASI")
    print("🤖 Menggunakan Algoritma Fuzzy K-Nearest Neighbors")
    print("=" * 80)
    
    # Load and process data
    academic_df, prestasi_summary = load_and_process_data()
    
    if academic_df is None or prestasi_summary is None:
        print("❌ Gagal memuat data. Pastikan file CSV tersedia.")
        return
    
    # Combine datasets
    combined_df = create_combined_dataset(academic_df, prestasi_summary)
    
    # Show dataset statistics
    print("\n" + "=" * 60)
    print("STATISTIK DATASET")
    print("=" * 60)
    print(combined_df.describe())
    
    # Visualize data
    visualize_data(combined_df)
    
    # Run classification experiment
    results = run_fuzzy_knn_experiment(combined_df)
    
    print("\n" + "=" * 60)
    print("KESIMPULAN")
    print("=" * 60)
    print("✅ Eksperimen Fuzzy K-NN berhasil diselesaikan!")
    print("📊 Dataset berhasil diproses dan diklasifikasikan")
    print("🎯 Model dapat mengidentifikasi mahasiswa berprestasi berdasarkan")
    print("   kombinasi prestasi akademik dan non-akademik")

if __name__ == "__main__":
    main()
