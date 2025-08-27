import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import webbrowser
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.mahasiswa_df = None
        self.prestasi_df = None
        self.combined_df = None
        self.cleaning_log = []
        
    def load_data(self, mahasiswa_file='mahasiswa_data_20250826_152655.csv', prestasi_file='prestasi.csv'):
        """Load both datasets"""
        try:
            self.mahasiswa_df = pd.read_csv(mahasiswa_file)
            self.prestasi_df = pd.read_csv(prestasi_file)
            
            print(f"✅ Data mahasiswa loaded: {len(self.mahasiswa_df)} records")
            print(f"✅ Data prestasi loaded: {len(self.prestasi_df)} records")
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_data_quality(self):
        """Analyze data quality and identify issues"""
        issues = {
            'mahasiswa_issues': [],
            'prestasi_issues': [],
            'consistency_issues': []
        }
        
        if self.mahasiswa_df is not None:
            # Check mahasiswa data issues
            missing_nim = self.mahasiswa_df['nim'].isnull().sum()
            empty_nim = (self.mahasiswa_df['nim'] == '').sum()
            duplicate_nim = self.mahasiswa_df['nim'].duplicated().sum()
            
            if missing_nim > 0:
                issues['mahasiswa_issues'].append(f"Missing NIM: {missing_nim} records")
            if empty_nim > 0:
                issues['mahasiswa_issues'].append(f"Empty NIM: {empty_nim} records")
            if duplicate_nim > 0:
                issues['mahasiswa_issues'].append(f"Duplicate NIM: {duplicate_nim} records")
                
            # Check for missing IPK data
            no_ipk_data = 0
            for idx, row in self.mahasiswa_df.iterrows():
                has_ipk = False
                for sem in range(1, 16):
                    ipk_col = f'khs{sem}_ipk'
                    if pd.notna(row.get(ipk_col)):
                        has_ipk = True
                        break
                if not has_ipk:
                    no_ipk_data += 1
            
            if no_ipk_data > 0:
                issues['mahasiswa_issues'].append(f"No IPK data: {no_ipk_data} students")
        
        if self.prestasi_df is not None:
            # Check prestasi data issues
            missing_id = self.prestasi_df['id_mahasiswa'].isnull().sum()
            empty_id = (self.prestasi_df['id_mahasiswa'] == '').sum()
            
            if missing_id > 0:
                issues['prestasi_issues'].append(f"Missing ID mahasiswa: {missing_id} records")
            if empty_id > 0:
                issues['prestasi_issues'].append(f"Empty ID mahasiswa: {empty_id} records")
        
        if self.mahasiswa_df is not None and self.prestasi_df is not None:
            # Check consistency between datasets
            valid_nims = set(self.mahasiswa_df['nim'].dropna().astype(str))
            prestasi_ids = set(self.prestasi_df['id_mahasiswa'].dropna().astype(str))
            
            unmatched_prestasi = len(prestasi_ids - valid_nims)
            if unmatched_prestasi > 0:
                issues['consistency_issues'].append(f"Prestasi with unmatched NIM: {unmatched_prestasi} unique IDs")
        
        return issues
    
    def clean_data(self):
        """Clean and prepare data for analysis"""
        print("\n" + "="*60)
        print("🧹 STARTING DATA CLEANING PROCESS")
        print("="*60)
        
        cleaning_stats = {
            'original_mahasiswa': len(self.mahasiswa_df) if self.mahasiswa_df is not None else 0,
            'original_prestasi': len(self.prestasi_df) if self.prestasi_df is not None else 0,
            'cleaned_mahasiswa': 0,
            'cleaned_prestasi': 0,
            'removed_records': 0
        }
        
        # Clean mahasiswa data
        if self.mahasiswa_df is not None:
            print("📊 Cleaning mahasiswa data...")
            
            # Remove records with missing or empty NIM
            original_count = len(self.mahasiswa_df)
            self.mahasiswa_df = self.mahasiswa_df.dropna(subset=['nim'])
            self.mahasiswa_df = self.mahasiswa_df[self.mahasiswa_df['nim'] != '']
            
            # Remove duplicates
            self.mahasiswa_df = self.mahasiswa_df.drop_duplicates(subset=['nim'])
            
            # Convert NIM to string for consistency
            self.mahasiswa_df['nim'] = self.mahasiswa_df['nim'].astype(str)
            
            cleaned_count = len(self.mahasiswa_df)
            removed = original_count - cleaned_count
            
            cleaning_stats['cleaned_mahasiswa'] = cleaned_count
            cleaning_stats['removed_records'] += removed
            
            print(f"  ✅ Removed {removed} problematic mahasiswa records")
            print(f"  ✅ Clean mahasiswa data: {cleaned_count} records")
        
        # Clean prestasi data
        if self.prestasi_df is not None:
            print("🏆 Cleaning prestasi data...")
            
            original_count = len(self.prestasi_df)
            
            # Remove records with missing or empty id_mahasiswa
            self.prestasi_df = self.prestasi_df.dropna(subset=['id_mahasiswa'])
            self.prestasi_df = self.prestasi_df[self.prestasi_df['id_mahasiswa'] != '']
            
            # Convert id_mahasiswa to string for consistency
            self.prestasi_df['id_mahasiswa'] = self.prestasi_df['id_mahasiswa'].astype(str)
            
            # Only keep prestasi data for students that exist in mahasiswa data
            if self.mahasiswa_df is not None:
                valid_nims = set(self.mahasiswa_df['nim'])
                self.prestasi_df = self.prestasi_df[
                    self.prestasi_df['id_mahasiswa'].isin(valid_nims)
                ]
            
            cleaned_count = len(self.prestasi_df)
            removed = original_count - cleaned_count
            
            cleaning_stats['cleaned_prestasi'] = cleaned_count
            cleaning_stats['removed_records'] += removed
            
            print(f"  ✅ Removed {removed} problematic prestasi records")
            print(f"  ✅ Clean prestasi data: {cleaned_count} records")
        
        # Save cleaning log
        self.cleaning_log.append({
            'timestamp': datetime.now().isoformat(),
            'stats': cleaning_stats
        })
        
        print(f"\n🎯 CLEANING SUMMARY:")
        print(f"  Total records removed: {cleaning_stats['removed_records']}")
        print(f"  Data quality improvement: {((cleaning_stats['cleaned_mahasiswa'] + cleaning_stats['cleaned_prestasi']) / (cleaning_stats['original_mahasiswa'] + cleaning_stats['original_prestasi']) * 100):.1f}%")
        
        return cleaning_stats
    
    def create_features(self):
        """Create features for classification"""
        print("\n" + "="*60)
        print("🔧 CREATING FEATURES")
        print("="*60)
        
        # Extract academic features
        academic_features = []
        
        for idx, row in self.mahasiswa_df.iterrows():
            nim = row['nim']
            
            # Extract semester data
            semester_data = []
            for sem in range(1, 16):
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
                stability_score = 1 / (1 + ips_std)
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
        prestasi_summary = self.prestasi_df.groupby('id_mahasiswa').agg({
            'tingkat': lambda x: len(x),
            'kategori': lambda x: (x == 'akademik').sum(),
            'jenis_prestasi': lambda x: (x == 'individu').sum()
        }).reset_index()
        
        prestasi_summary.columns = ['nim', 'total_prestasi', 'prestasi_akademik', 'prestasi_individu']
        prestasi_summary['prestasi_non_akademik'] = prestasi_summary['total_prestasi'] - prestasi_summary['prestasi_akademik']
        
        # Calculate achievement levels
        tingkat_scores = self.prestasi_df.groupby('id_mahasiswa')['tingkat'].apply(
            lambda x: sum([3 if t == 'internasional' else 2 if t == 'nasional' else 1 for t in x])
        ).reset_index()
        tingkat_scores.columns = ['nim', 'achievement_level_score']
        
        prestasi_summary = prestasi_summary.merge(tingkat_scores, on='nim', how='left')
        prestasi_summary['achievement_level_score'] = prestasi_summary['achievement_level_score'].fillna(0)
        
        # Merge datasets
        self.combined_df = academic_df.merge(prestasi_summary, on='nim', how='left')
        
        # Fill missing values
        achievement_cols = ['total_prestasi', 'prestasi_akademik', 'prestasi_non_akademik', 
                           'prestasi_individu', 'achievement_level_score']
        for col in achievement_cols:
            self.combined_df[col] = self.combined_df[col].fillna(0)
        
        # Create composite scores
        self.combined_df['academic_score'] = (
            self.combined_df['final_ipk'] * 0.4 + 
            self.combined_df['stability_score'] * 0.3 + 
            (self.combined_df['academic_trend'] + 1) * 0.3
        )
        
        self.combined_df['achievement_score'] = (
            self.combined_df['total_prestasi'] * 0.3 +
            self.combined_df['achievement_level_score'] * 0.4 +
            self.combined_df['prestasi_akademik'] * 0.3
        )
        
        # Create labels
        def create_label(row):
            academic_threshold = row['final_ipk'] >= 3.50 and row['stability_score'] >= 0.5
            achievement_threshold = row['total_prestasi'] >= 2 or row['achievement_level_score'] >= 3
            
            if academic_threshold and achievement_threshold:
                return 1
            elif row['final_ipk'] >= 3.80 or row['total_prestasi'] >= 4:
                return 1
            else:
                return 0
        
        self.combined_df['label'] = self.combined_df.apply(create_label, axis=1)
        
        print(f"✅ Combined dataset created: {len(self.combined_df)} records")
        print(f"✅ Berprestasi: {sum(self.combined_df['label'])} ({sum(self.combined_df['label'])/len(self.combined_df)*100:.1f}%)")
        
        return self.combined_df
    
    def export_clean_data(self, output_dir='clean_data'):
        """Export cleaned data to CSV files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.mahasiswa_df is not None:
            mahasiswa_file = f"{output_dir}/mahasiswa_clean_{timestamp}.csv"
            self.mahasiswa_df.to_csv(mahasiswa_file, index=False)
            print(f"✅ Clean mahasiswa data exported to: {mahasiswa_file}")
        
        if self.prestasi_df is not None:
            prestasi_file = f"{output_dir}/prestasi_clean_{timestamp}.csv"
            self.prestasi_df.to_csv(prestasi_file, index=False)
            print(f"✅ Clean prestasi data exported to: {prestasi_file}")
        
        if self.combined_df is not None:
            combined_file = f"{output_dir}/combined_data_{timestamp}.csv"
            self.combined_df.to_csv(combined_file, index=False)
            print(f"✅ Combined dataset exported to: {combined_file}")
        
        # Export cleaning log
        log_file = f"{output_dir}/cleaning_log_{timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump(self.cleaning_log, f, indent=2)
        print(f"✅ Cleaning log exported to: {log_file}")
    
    def generate_summary_report(self):
        """Generate a summary report of the data processing"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'mahasiswa_records': len(self.mahasiswa_df) if self.mahasiswa_df is not None else 0,
                'prestasi_records': len(self.prestasi_df) if self.prestasi_df is not None else 0,
                'combined_records': len(self.combined_df) if self.combined_df is not None else 0
            },
            'quality_metrics': {},
            'feature_summary': {}
        }
        
        if self.combined_df is not None:
            report['quality_metrics'] = {
                'berprestasi_count': int(sum(self.combined_df['label'])),
                'berprestasi_percentage': float(sum(self.combined_df['label']) / len(self.combined_df) * 100),
                'avg_ipk': float(self.combined_df['final_ipk'].mean()),
                'avg_prestasi_count': float(self.combined_df['total_prestasi'].mean())
            }
            
            report['feature_summary'] = {
                'total_features': len(self.combined_df.columns) - 1,  # excluding nim
                'academic_features': 7,
                'achievement_features': 5,
                'composite_features': 2
            }
        
        return report

def open_dashboard():
    """Open the dashboard in the default web browser"""
    dashboard_path = os.path.abspath('dashboard.html')
    if os.path.exists(dashboard_path):
        webbrowser.open(f'file://{dashboard_path}')
        print(f"🌐 Dashboard opened: {dashboard_path}")
    else:
        print("❌ Dashboard file not found!")

def main():
    """Main function to run data processing and open dashboard"""
    print("🎓 SISTEM KLASIFIKASI MAHASISWA BERPRESTASI")
    print("🧹 Data Processing & Dashboard")
    print("=" * 80)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load data
    if not processor.load_data():
        print("❌ Failed to load data. Please check if CSV files exist.")
        return
    
    # Analyze data quality
    print("\n🔍 ANALYZING DATA QUALITY...")
    issues = processor.analyze_data_quality()
    
    for category, problems in issues.items():
        if problems:
            print(f"\n⚠️  {category.replace('_', ' ').title()}:")
            for problem in problems:
                print(f"  - {problem}")
    
    # Clean data
    cleaning_stats = processor.clean_data()
    
    # Create features
    combined_df = processor.create_features()
    
    # Export clean data
    print("\n💾 EXPORTING CLEAN DATA...")
    processor.export_clean_data()
    
    # Generate summary report
    report = processor.generate_summary_report()
    
    print("\n📊 FINAL SUMMARY:")
    print(f"  Total mahasiswa: {report['data_summary']['mahasiswa_records']}")
    print(f"  Total prestasi: {report['data_summary']['prestasi_records']}")
    print(f"  Combined dataset: {report['data_summary']['combined_records']}")
    print(f"  Mahasiswa berprestasi: {report['quality_metrics']['berprestasi_count']} ({report['quality_metrics']['berprestasi_percentage']:.1f}%)")
    print(f"  Rata-rata IPK: {report['quality_metrics']['avg_ipk']:.2f}")
    
    # Open dashboard
    print("\n🌐 OPENING DASHBOARD...")
    open_dashboard()
    
    print("\n✅ PROCESS COMPLETED!")
    print("📊 You can now use the dashboard to explore the data interactively.")
    print("🤖 Ready for Fuzzy K-NN classification!")

if __name__ == "__main__":
    main()
