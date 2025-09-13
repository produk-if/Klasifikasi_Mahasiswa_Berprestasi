import pandas as pd
import numpy as np
import json
import uuid
import random
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataProcessor:
    """
    Comprehensive data processing system for student achievement classification
    that integrates academic records with organizational activities data
    """
    
    def __init__(self):
        self.mahasiswa_df = None
        self.prestasi_df = None
        self.organizational_df = None
        self.combined_df = None
        self.cleaning_log = []
        self.feature_weights = {
            'academic': 0.40,    # Academic performance (IPK, stability, trend)
            'achievement': 0.35,  # Achievements (competitions, publications, awards)
            'organizational': 0.25 # Organizational involvement (leadership, duration, diversity)
        }
        
    def load_data(self, mahasiswa_file='mahasiswa_data_20250826_152655.csv', prestasi_file='prestasi.csv', organizational_file=None):
        """Load multiple data sources with validation"""
        try:
            # Load academic records
            self.mahasiswa_df = pd.read_csv(mahasiswa_file)
            print(f"✅ Academic records loaded: {len(self.mahasiswa_df)} students")
            
            # Load achievement records  
            self.prestasi_df = pd.read_csv(prestasi_file)
            print(f"✅ Achievement records loaded: {len(self.prestasi_df)} records")
            
            # Load or create organizational activities data
            if organizational_file and os.path.exists(organizational_file):
                self.organizational_df = pd.read_csv(organizational_file)
                print(f"✅ Organizational data loaded: {len(self.organizational_df)} records")
            else:
                print("⚠️  No organizational data file found. Creating synthetic organizational activities...")
                self._create_synthetic_organizational_data()
            
            # Log data loading
            self._log_action("data_loading", {
                'mahasiswa_count': len(self.mahasiswa_df),
                'prestasi_count': len(self.prestasi_df),
                'organizational_count': len(self.organizational_df) if self.organizational_df is not None else 0,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self._log_action("error", {'error': str(e), 'function': 'load_data'})
            return False
    
    def _create_synthetic_organizational_data(self):
        """Create synthetic organizational activities data if LPKA data is unavailable"""
        if self.mahasiswa_df is None:
            return
        
        # Organization types
        org_types = ['academic', 'social', 'religious', 'sports', 'arts', 'technology', 'volunteer']
        org_names = {
            'academic': ['BEM Fakultas', 'Himpunan Mahasiswa Prodi', 'ORMAWA', 'Unit Kegiatan Mahasiswa'],
            'social': ['Karang Taruna', 'Komunitas Peduli Sosial', 'Relawan Bencana', 'Tim Bakti Sosial'],
            'religious': ['Kerohanian Islam', 'Persekutuan Kristen', 'Dharma Wacana', 'Kegiatan Keagamaan'],
            'sports': ['Unit Olahraga', 'Tim Futsal', 'Basket Club', 'Badminton Club'],
            'arts': ['Unit Kesenian', 'Teater Kampus', 'Paduan Suara', 'Tari Tradisional'],
            'technology': ['IT Club', 'Robotika', 'Programming Community', 'Tech Startup'],
            'volunteer': ['PMI', 'Pramuka', 'KKN', 'Volunteer Community']
        }
        
        organizational_data = []
        np.random.seed(42)  # For reproducible synthetic data
        
        for _, student in self.mahasiswa_df.iterrows():
            nim = student['nim']
            
            # 70% of students have organizational involvement
            if np.random.random() < 0.7:
                # Number of organizational activities (1-4)
                num_activities = np.random.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.2, 0.05])
                
                selected_types = np.random.choice(org_types, size=min(num_activities, len(org_types)), replace=False)
                
                for org_type in selected_types:
                    org_name = np.random.choice(org_names[org_type])
                    
                    # Role (30% leadership, 70% member)
                    is_leader = np.random.random() < 0.3
                    role = np.random.choice(['Ketua', 'Wakil Ketua', 'Sekretaris', 'Bendahara', 'Koordinator']) if is_leader else 'Anggota'
                    
                    # Duration (1-8 semesters)
                    duration_semesters = np.random.randint(1, 9)
                    
                    # Start year based on student's enrollment year
                    start_year = int(student['angkatan']) + np.random.randint(0, 3)
                    
                    organizational_data.append({
                        'nim': nim,
                        'organization_name': org_name,
                        'organization_type': org_type,
                        'role': role,
                        'is_leadership': is_leader,
                        'start_year': start_year,
                        'duration_semesters': duration_semesters,
                        'activity_level': np.random.choice(['active', 'very_active', 'moderately_active'], p=[0.5, 0.3, 0.2])
                    })
        
        self.organizational_df = pd.DataFrame(organizational_data)
        print(f"✅ Synthetic organizational data created: {len(self.organizational_df)} activity records")
        
        # Save synthetic data for future use
        self.organizational_df.to_csv('synthetic_organizational_activities.csv', index=False)
        print("💾 Synthetic organizational data saved as 'synthetic_organizational_activities.csv'")
    
    def analyze_data_quality(self):
        """Comprehensive data quality analysis"""
        issues = {
            'mahasiswa_issues': [],
            'prestasi_issues': [],
            'organizational_issues': [],
            'consistency_issues': []
        }
        
        if self.mahasiswa_df is not None:
            # Academic data quality checks
            missing_nim = self.mahasiswa_df['nim'].isnull().sum()
            empty_nim = (self.mahasiswa_df['nim'].astype(str).str.strip() == '').sum()
            duplicate_nim = self.mahasiswa_df['nim'].duplicated().sum()
            
            if missing_nim > 0:
                issues['mahasiswa_issues'].append(f"Missing NIM: {missing_nim} records")
            if empty_nim > 0:
                issues['mahasiswa_issues'].append(f"Empty NIM: {empty_nim} records")
            if duplicate_nim > 0:
                issues['mahasiswa_issues'].append(f"Duplicate NIM: {duplicate_nim} records")
                
            # Check for students with no IPK data
            students_no_ipk = []
            for idx, row in self.mahasiswa_df.iterrows():
                has_ipk = False
                for sem in range(1, 16):
                    ipk_col = f'khs{sem}_ipk'
                    if pd.notna(row.get(ipk_col)) and row.get(ipk_col) > 0:
                        has_ipk = True
                        break
                if not has_ipk:
                    students_no_ipk.append(row['nim'])
            
            if students_no_ipk:
                issues['mahasiswa_issues'].append(f"No valid IPK data: {len(students_no_ipk)} students")
        
        if self.prestasi_df is not None:
            # Achievement data quality checks
            missing_id = self.prestasi_df['id_mahasiswa'].isnull().sum()
            empty_id = (self.prestasi_df['id_mahasiswa'].astype(str).str.strip() == '').sum()
            
            if missing_id > 0:
                issues['prestasi_issues'].append(f"Missing ID mahasiswa: {missing_id} records")
            if empty_id > 0:
                issues['prestasi_issues'].append(f"Empty ID mahasiswa: {empty_id} records")
            
            # Check for invalid achievement levels
            valid_tingkat = ['lokal', 'regional', 'nasional', 'internasional']
            invalid_tingkat = self.prestasi_df[~self.prestasi_df['tingkat'].str.lower().isin(valid_tingkat)]
            if len(invalid_tingkat) > 0:
                issues['prestasi_issues'].append(f"Invalid achievement levels: {len(invalid_tingkat)} records")
        
        if self.organizational_df is not None:
            # Organizational data quality checks
            missing_nim_org = self.organizational_df['nim'].isnull().sum()
            if missing_nim_org > 0:
                issues['organizational_issues'].append(f"Missing NIM in organizational data: {missing_nim_org} records")
        
        # Cross-dataset consistency checks
        if self.mahasiswa_df is not None and self.prestasi_df is not None:
            valid_nims = set(self.mahasiswa_df['nim'].dropna().astype(str))
            prestasi_ids = set(self.prestasi_df['id_mahasiswa'].dropna().astype(str))
            
            # Check for prestasi records with unmatched NIMs
            unmatched_prestasi = prestasi_ids - valid_nims
            if unmatched_prestasi:
                issues['consistency_issues'].append(f"Prestasi with unmatched NIM: {len(unmatched_prestasi)} unique IDs")
                # Don't remove these - they might be valid with different format
        
        return issues
    
    def clean_data(self):
        """Enhanced data cleaning that preserves achievement records"""
        print("\n" + "="*80)
        print("🧹 ENHANCED DATA CLEANING PROCESS")
        print("="*80)
        
        cleaning_stats = {
            'original_mahasiswa': len(self.mahasiswa_df) if self.mahasiswa_df is not None else 0,
            'original_prestasi': len(self.prestasi_df) if self.prestasi_df is not None else 0,
            'original_organizational': len(self.organizational_df) if self.organizational_df is not None else 0
        }
        
        # Step 1: Clean academic data
        print("\n📊 Step 1: Cleaning Academic Data")
        if self.mahasiswa_df is not None:
            initial_count = len(self.mahasiswa_df)
            
            # Remove duplicate NIMs (keep first occurrence)
            self.mahasiswa_df = self.mahasiswa_df.drop_duplicates(subset=['nim'], keep='first')
            
            # Clean NIM format (remove spaces, ensure string format)
            self.mahasiswa_df['nim'] = self.mahasiswa_df['nim'].astype(str).str.strip()
            
            # Remove records with completely empty NIM
            self.mahasiswa_df = self.mahasiswa_df[self.mahasiswa_df['nim'].notna()]
            self.mahasiswa_df = self.mahasiswa_df[self.mahasiswa_df['nim'] != '']
            self.mahasiswa_df = self.mahasiswa_df[self.mahasiswa_df['nim'] != 'nan']
            
            print(f"   Academic records: {initial_count} → {len(self.mahasiswa_df)}")
            cleaning_stats['cleaned_mahasiswa'] = len(self.mahasiswa_df)
        
        # Step 2: Clean achievement data (PRESERVE RECORDS)
        print("\n🏆 Step 2: Cleaning Achievement Data (Preserving Records)")
        if self.prestasi_df is not None:
            initial_count = len(self.prestasi_df)
            
            # Clean ID mahasiswa format
            self.prestasi_df['id_mahasiswa'] = self.prestasi_df['id_mahasiswa'].astype(str).str.strip()
            
            # Instead of removing unmatched records, try to match them
            valid_nims = set(self.mahasiswa_df['nim'].astype(str)) if self.mahasiswa_df is not None else set()
            prestasi_ids = set(self.prestasi_df['id_mahasiswa'].astype(str))
            
            # Keep all prestasi records, but flag unmatched ones
            self.prestasi_df['nim_matched'] = self.prestasi_df['id_mahasiswa'].isin(valid_nims)
            unmatched_count = len(self.prestasi_df[~self.prestasi_df['nim_matched']])
            
            if unmatched_count > 0:
                print(f"   ⚠️  {unmatched_count} achievement records have unmatched NIMs (keeping for manual review)")
            
            # Standardize achievement levels
            tingkat_mapping = {
                'lokal': 'lokal',
                'regional': 'regional', 
                'nasional': 'nasional',
                'internasional': 'internasional',
                'daerah': 'regional',
                'wilayah': 'regional',
                'kota': 'lokal',
                'kabupaten': 'lokal',
                'provinsi': 'regional'
            }
            
            self.prestasi_df['tingkat_cleaned'] = self.prestasi_df['tingkat'].str.lower().map(tingkat_mapping)
            self.prestasi_df['tingkat_cleaned'].fillna(self.prestasi_df['tingkat'], inplace=True)
            
            print(f"   Achievement records: {initial_count} → {len(self.prestasi_df)} (preserved)")
            cleaning_stats['cleaned_prestasi'] = len(self.prestasi_df)
        
        # Step 3: Clean organizational data
        print("\n🏢 Step 3: Cleaning Organizational Data")
        if self.organizational_df is not None:
            initial_count = len(self.organizational_df)
            
            # Clean NIM format
            self.organizational_df['nim'] = self.organizational_df['nim'].astype(str).str.strip()
            
            # Remove records with missing NIM
            self.organizational_df = self.organizational_df[self.organizational_df['nim'].notna()]
            self.organizational_df = self.organizational_df[self.organizational_df['nim'] != '']
            
            print(f"   Organizational records: {initial_count} → {len(self.organizational_df)}")
            cleaning_stats['cleaned_organizational'] = len(self.organizational_df)
        
        # Log cleaning results
        self._log_action("data_cleaning", {
            'cleaning_stats': cleaning_stats,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"\n✅ Data cleaning completed successfully!")
        print(f"📋 Summary: {cleaning_stats}")
        return cleaning_stats
    
    def engineer_features(self):
        """Advanced feature engineering including organizational activities"""
        print("\n" + "="*80)
        print("🔧 ADVANCED FEATURE ENGINEERING")
        print("="*80)
        
        if self.mahasiswa_df is None:
            print("❌ No academic data available for feature engineering")
            return None
        
        features_list = []
        
        for _, student in self.mahasiswa_df.iterrows():
            nim = student['nim']
            features = {'nim': nim}
            
            # 1. ACADEMIC PERFORMANCE FEATURES (40%)
            print(f"Processing academic features for student {nim}...")
            academic_features = self._extract_academic_features(student)
            features.update(academic_features)
            
            # 2. ACHIEVEMENT FEATURES (35%)
            achievement_features = self._extract_achievement_features(nim)
            features.update(achievement_features)
            
            # 3. ORGANIZATIONAL FEATURES (25%)
            org_features = self._extract_organizational_features(nim)
            features.update(org_features)
            
            # 4. COMPOSITE SCORES
            composite_scores = self._calculate_composite_scores(academic_features, achievement_features, org_features)
            features.update(composite_scores)
            
            features_list.append(features)
        
        # Create features DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Generate balanced labels
        features_df = self._generate_balanced_labels(features_df)
        
        print(f"✅ Feature engineering completed for {len(features_df)} students")
        return features_df
    
    def _extract_academic_features(self, student):
        """Extract comprehensive academic performance features"""
        features = {}
        
        # Collect all IPK values
        ipk_values = []
        ips_values = []
        
        for sem in range(1, 16):
            ipk = student.get(f'khs{sem}_ipk')
            ips = student.get(f'khs{sem}_ips')
            
            if pd.notna(ipk) and ipk > 0:
                ipk_values.append(float(ipk))
            if pd.notna(ips) and ips > 0:
                ips_values.append(float(ips))
        
        if ipk_values:
            # Basic academic metrics
            features['final_ipk'] = ipk_values[-1]
            features['avg_ipk'] = np.mean(ipk_values)
            features['max_ipk'] = np.max(ipk_values)
            features['min_ipk'] = np.min(ipk_values)
            
            # Academic stability (lower std = more stable)
            features['ipk_stability'] = 1 / (1 + np.std(ipk_values))  # Normalized stability score
            
            # Academic trend (improvement over time)
            if len(ipk_values) > 1:
                # Linear trend using least squares
                x = np.arange(len(ipk_values))
                trend_coef = np.polyfit(x, ipk_values, 1)[0]
                features['ipk_trend'] = max(0, trend_coef)  # Positive trend only
            else:
                features['ipk_trend'] = 0
                
            # Performance consistency
            features['performance_consistency'] = 1 - (np.std(ipk_values) / np.mean(ipk_values))
            
        else:
            # Default values if no IPK data
            for key in ['final_ipk', 'avg_ipk', 'max_ipk', 'min_ipk', 'ipk_stability', 'ipk_trend', 'performance_consistency']:
                features[key] = 0
        
        # Study duration analysis
        if pd.notna(student.get('masaStudi')):
            masa_studi = student['masaStudi']
            # Extract years from format like "4 Tahun, 0 Bulan"
            if 'Tahun' in str(masa_studi):
                years = float(str(masa_studi).split('Tahun')[0].strip())
                features['study_duration_years'] = years
                features['on_time_graduation'] = 1 if years <= 4 else 0
            else:
                features['study_duration_years'] = 4
                features['on_time_graduation'] = 1
        else:
            features['study_duration_years'] = 4
            features['on_time_graduation'] = 1
        
        return features
    
    def _extract_achievement_features(self, nim):
        """Extract achievement-based features"""
        features = {}
        
        if self.prestasi_df is None:
            # Default values if no achievement data
            feature_names = [
                'total_achievements', 'akademik_achievements', 'non_akademik_achievements',
                'lokal_achievements', 'regional_achievements', 'nasional_achievements', 'internasional_achievements',
                'achievement_diversity_score', 'recent_achievements', 'achievement_frequency',
                'international_recognition', 'publication_count', 'competition_wins'
            ]
            for name in feature_names:
                features[name] = 0
            return features
        
        # Filter achievements for this student
        student_prestasi = self.prestasi_df[
            (self.prestasi_df['id_mahasiswa'].astype(str) == str(nim)) |
            (self.prestasi_df['nim_matched'] == True)
        ]
        
        # Basic achievement counts
        features['total_achievements'] = len(student_prestasi)
        
        # Achievement by category
        akademik_count = len(student_prestasi[student_prestasi['kategori'].str.lower() == 'akademik'])
        non_akademik_count = len(student_prestasi[student_prestasi['kategori'].str.lower() == 'non_akademik'])
        
        features['akademik_achievements'] = akademik_count
        features['non_akademik_achievements'] = non_akademik_count
        
        # Achievement by level
        tingkat_counts = student_prestasi['tingkat_cleaned'].value_counts() if 'tingkat_cleaned' in student_prestasi.columns else student_prestasi['tingkat'].value_counts()
        
        features['lokal_achievements'] = tingkat_counts.get('lokal', 0)
        features['regional_achievements'] = tingkat_counts.get('regional', 0) 
        features['nasional_achievements'] = tingkat_counts.get('nasional', 0)
        features['internasional_achievements'] = tingkat_counts.get('internasional', 0)
        
        # Achievement diversity (variety of categories and levels)
        unique_categories = student_prestasi['kategori'].nunique() if len(student_prestasi) > 0 else 0
        unique_levels = student_prestasi['tingkat'].nunique() if len(student_prestasi) > 0 else 0
        features['achievement_diversity_score'] = unique_categories + unique_levels
        
        # Recent achievements (last 2 years)
        if len(student_prestasi) > 0 and 'tanggal' in student_prestasi.columns:
            try:
                student_prestasi['tanggal_parsed'] = pd.to_datetime(student_prestasi['tanggal'], errors='coerce')
                recent_cutoff = datetime(2022, 1, 1)  # Adjust based on your data timeframe
                recent_achievements = student_prestasi[student_prestasi['tanggal_parsed'] >= recent_cutoff]
                features['recent_achievements'] = len(recent_achievements)
            except:
                features['recent_achievements'] = len(student_prestasi)  # Default to all if parsing fails
        else:
            features['recent_achievements'] = 0
        
        # Achievement frequency (achievements per year of study)
        total_years = features.get('study_duration_years', 4)
        features['achievement_frequency'] = features['total_achievements'] / max(1, total_years)
        
        # Special recognition flags
        features['international_recognition'] = 1 if features['internasional_achievements'] > 0 else 0
        features['publication_count'] = akademik_count  # Academic achievements often include publications
        features['competition_wins'] = non_akademik_count  # Non-academic often competitions
        
        return features
    
    def _extract_organizational_features(self, nim):
        """Extract organizational involvement features"""
        features = {}
        
        if self.organizational_df is None:
            # Default values if no organizational data
            feature_names = [
                'total_organizations', 'leadership_positions', 'member_positions',
                'leadership_diversity', 'total_org_duration', 'avg_org_duration',
                'academic_orgs', 'social_orgs', 'religious_orgs', 'sports_orgs',
                'arts_orgs', 'technology_orgs', 'volunteer_orgs',
                'org_type_diversity', 'inter_org_collaboration', 'current_active_orgs'
            ]
            for name in feature_names:
                features[name] = 0
            return features
        
        # Filter organizational data for this student
        student_orgs = self.organizational_df[self.organizational_df['nim'].astype(str) == str(nim)]
        
        # Basic counts
        features['total_organizations'] = len(student_orgs)
        
        # Leadership vs membership
        leadership_count = len(student_orgs[student_orgs['is_leadership'] == True])
        member_count = len(student_orgs[student_orgs['is_leadership'] == False])
        
        features['leadership_positions'] = leadership_count
        features['member_positions'] = member_count
        
        # Leadership diversity (different types of organizations led)
        if leadership_count > 0:
            leadership_orgs = student_orgs[student_orgs['is_leadership'] == True]
            features['leadership_diversity'] = leadership_orgs['organization_type'].nunique()
        else:
            features['leadership_diversity'] = 0
        
        # Duration analysis
        if len(student_orgs) > 0:
            durations = student_orgs['duration_semesters'].tolist()
            features['total_org_duration'] = sum(durations)
            features['avg_org_duration'] = np.mean(durations)
        else:
            features['total_org_duration'] = 0
            features['avg_org_duration'] = 0
        
        # Organization types
        org_type_counts = student_orgs['organization_type'].value_counts()
        org_types = ['academic', 'social', 'religious', 'sports', 'arts', 'technology', 'volunteer']
        
        for org_type in org_types:
            features[f'{org_type}_orgs'] = org_type_counts.get(org_type, 0)
        
        # Organizational diversity
        features['org_type_diversity'] = student_orgs['organization_type'].nunique()
        
        # Inter-organizational collaboration (approximate)
        features['inter_org_collaboration'] = min(3, len(student_orgs) - 1) if len(student_orgs) > 1 else 0
        
        # Current active organizations (estimate based on recent start years)
        current_year = 2024  # Adjust based on current context
        recent_orgs = student_orgs[student_orgs['start_year'] >= current_year - 2]
        features['current_active_orgs'] = len(recent_orgs)
        
        return features
    
    def _calculate_composite_scores(self, academic_features, achievement_features, org_features):
        """Calculate weighted composite scores"""
        scores = {}
        
        # Academic Performance Score (40%)
        academic_score = 0
        if academic_features.get('final_ipk', 0) > 0:
            # IPK component (0-1 scale)
            ipk_score = min(1.0, academic_features['final_ipk'] / 4.0)
            
            # Stability component (0-1 scale) 
            stability_score = academic_features.get('ipk_stability', 0)
            
            # Trend component (0-1 scale)
            trend_score = min(1.0, max(0, academic_features.get('ipk_trend', 0)))
            
            # On-time graduation bonus
            graduation_bonus = 0.1 if academic_features.get('on_time_graduation', 0) else 0
            
            academic_score = (0.6 * ipk_score + 0.3 * stability_score + 0.1 * trend_score) + graduation_bonus
        
        scores['academic_performance_score'] = min(1.0, academic_score)
        
        # Achievement Score (35%)
        achievement_score = 0
        total_achievements = achievement_features.get('total_achievements', 0)
        
        if total_achievements > 0:
            # Achievement count component
            count_score = min(1.0, total_achievements / 5.0)  # Normalized by 5 achievements
            
            # Level weighting (international > national > regional > local)
            level_score = (
                achievement_features.get('internasional_achievements', 0) * 1.0 +
                achievement_features.get('nasional_achievements', 0) * 0.8 +
                achievement_features.get('regional_achievements', 0) * 0.6 +
                achievement_features.get('lokal_achievements', 0) * 0.4
            ) / max(1, total_achievements)
            
            # Diversity bonus
            diversity_bonus = min(0.2, achievement_features.get('achievement_diversity_score', 0) * 0.05)
            
            achievement_score = 0.5 * count_score + 0.4 * level_score + 0.1 + diversity_bonus
        
        scores['achievement_score'] = min(1.0, achievement_score)
        
        # Organizational Involvement Score (25%)
        org_score = 0
        total_orgs = org_features.get('total_organizations', 0)
        
        if total_orgs > 0:
            # Leadership component
            leadership_score = min(1.0, org_features.get('leadership_positions', 0) / 2.0)
            
            # Duration component 
            duration_score = min(1.0, org_features.get('avg_org_duration', 0) / 6.0)  # 6 semesters max
            
            # Diversity component
            diversity_score = min(1.0, org_features.get('org_type_diversity', 0) / 4.0)  # 4 types max
            
            org_score = 0.5 * leadership_score + 0.3 * duration_score + 0.2 * diversity_score
        
        scores['organizational_involvement_score'] = min(1.0, org_score)
        
        # Overall Composite Score
        overall_score = (
            self.feature_weights['academic'] * scores['academic_performance_score'] +
            self.feature_weights['achievement'] * scores['achievement_score'] +
            self.feature_weights['organizational'] * scores['organizational_involvement_score']
        )
        
        scores['composite_score'] = overall_score
        
        return scores
    
    def _generate_balanced_labels(self, features_df):
        """Generate balanced labels using multiple criteria"""
        print("\n🏷️  Generating balanced labels using multiple criteria...")
        
        # Define criteria thresholds
        academic_criteria = (
            (features_df['final_ipk'] >= 3.5) & 
            (features_df['ipk_stability'] >= 0.6)
        )
        
        achievement_criteria = (
            (features_df['total_achievements'] >= 2) |
            (features_df['international_recognition'] == 1) |
            (features_df['nasional_achievements'] >= 1)
        )
        
        leadership_criteria = (
            (features_df['leadership_positions'] >= 1) |
            (features_df['total_organizations'] >= 2)
        )
        
        # Count how many criteria each student meets
        criteria_count = (
            academic_criteria.astype(int) +
            achievement_criteria.astype(int) + 
            leadership_criteria.astype(int)
        )
        
        # Label as "berprestasi" if meets 2 out of 3 criteria
        features_df['label'] = (criteria_count >= 2).astype(int)
        features_df['criteria_count'] = criteria_count
        
        # Additional high-performer category (meets all 3 criteria)
        features_df['high_performer'] = (criteria_count == 3).astype(int)
        
        # Label distribution
        label_dist = features_df['label'].value_counts()
        print(f"Label distribution:")
        print(f"  Non-berprestasi (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(features_df)*100:.1f}%)")
        print(f"  Berprestasi (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(features_df)*100:.1f}%)")
        
        high_performer_count = features_df['high_performer'].sum()
        print(f"  High performers: {high_performer_count} ({high_performer_count/len(features_df)*100:.1f}%)")
        
        return features_df
    
    def export_clean_data(self, output_dir='clean_data'):
        """Export cleaned datasets with organizational features"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate features
        features_df = self.engineer_features()
        if features_df is None:
            print("❌ Failed to generate features")
            return
        
        # Export files
        files_exported = []
        
        # 1. Clean academic data
        if self.mahasiswa_df is not None:
            academic_file = os.path.join(output_dir, f'mahasiswa_clean_{timestamp}.csv')
            self.mahasiswa_df.to_csv(academic_file, index=False)
            files_exported.append(academic_file)
        
        # 2. Clean achievement data (preserved)
        if self.prestasi_df is not None:
            prestasi_file = os.path.join(output_dir, f'prestasi_clean_{timestamp}.csv')
            self.prestasi_df.to_csv(prestasi_file, index=False)
            files_exported.append(prestasi_file)
        
        # 3. Organizational data
        if self.organizational_df is not None:
            org_file = os.path.join(output_dir, f'organizational_clean_{timestamp}.csv')
            self.organizational_df.to_csv(org_file, index=False)
            files_exported.append(org_file)
        
        # 4. Combined features dataset
        combined_file = os.path.join(output_dir, f'combined_features_{timestamp}.csv')
        features_df.to_csv(combined_file, index=False)
        files_exported.append(combined_file)
        
        # 5. Export processing log
        log_file = os.path.join(output_dir, f'processing_log_{timestamp}.json')
        
        # Generate summary statistics
        summary_stats = {
            'processing_timestamp': timestamp,
            'data_sources': {
                'academic_records': len(self.mahasiswa_df) if self.mahasiswa_df is not None else 0,
                'achievement_records': len(self.prestasi_df) if self.prestasi_df is not None else 0,
                'organizational_records': len(self.organizational_df) if self.organizational_df is not None else 0
            },
            'feature_summary': {
                'total_students': len(features_df),
                'berprestasi_count': int(features_df['label'].sum()),
                'berprestasi_percentage': float(features_df['label'].mean() * 100),
                'high_performer_count': int(features_df['high_performer'].sum()),
                'average_composite_score': float(features_df['composite_score'].mean())
            },
            'feature_weights': self.feature_weights,
            'files_exported': files_exported,
            'processing_log': self.cleaning_log
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Data export completed!")
        print(f"📁 Files exported to: {output_dir}/")
        for file_path in files_exported + [log_file]:
            print(f"   - {os.path.basename(file_path)}")
        
        print(f"\n📊 Processing Summary:")
        print(f"   - Total students processed: {len(features_df)}")
        print(f"   - Berprestasi students: {int(features_df['label'].sum())} ({features_df['label'].mean()*100:.1f}%)")
        print(f"   - High performers: {int(features_df['high_performer'].sum())} ({features_df['high_performer'].mean()*100:.1f}%)")
        print(f"   - Average composite score: {features_df['composite_score'].mean():.3f}")
        
        return files_exported, summary_stats
    
    def _log_action(self, action_type, details):
        """Log processing actions for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action_type,
            'details': details
        }
        self.cleaning_log.append(log_entry)
    
    def validate_processing(self):
        """Validate the processing results"""
        print("\n" + "="*80)
        print("✅ PROCESSING VALIDATION")
        print("="*80)
        
        validation_results = {
            'data_integrity': True,
            'feature_quality': True,
            'label_balance': True,
            'issues': []
        }
        
        # Check data integrity
        if self.mahasiswa_df is None:
            validation_results['data_integrity'] = False
            validation_results['issues'].append("No academic data loaded")
        
        # Validate features if available
        features_df = self.engineer_features()
        if features_df is not None:
            # Check for missing values in critical features
            critical_features = ['final_ipk', 'total_achievements', 'total_organizations', 'composite_score']
            for feature in critical_features:
                if feature in features_df.columns:
                    missing_count = features_df[feature].isnull().sum()
                    if missing_count > 0:
                        validation_results['feature_quality'] = False
                        validation_results['issues'].append(f"Missing values in {feature}: {missing_count}")
            
            # Check label balance
            if 'label' in features_df.columns:
                label_balance = features_df['label'].mean()
                if label_balance < 0.1 or label_balance > 0.9:
                    validation_results['label_balance'] = False
                    validation_results['issues'].append(f"Imbalanced labels: {label_balance:.3f}")
        
        # Print validation results
        if all(validation_results[key] for key in ['data_integrity', 'feature_quality', 'label_balance']):
            print("🎉 All validations passed!")
        else:
            print("⚠️  Validation issues found:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")
        
        return validation_results


def main():
    """Main execution function"""
    print("🚀 Enhanced Data Processing System for Student Achievement Classification")
    print("="*80)
    
    # Initialize processor
    processor = EnhancedDataProcessor()
    
    # Load data
    if not processor.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Analyze data quality
    print("\n📊 Data Quality Analysis:")
    issues = processor.analyze_data_quality()
    for category, issue_list in issues.items():
        if issue_list:
            print(f"\n{category.replace('_', ' ').title()}:")
            for issue in issue_list:
                print(f"  - {issue}")
    
    # Clean data
    cleaning_stats = processor.clean_data()
    
    # Export clean data with features
    files_exported, summary = processor.export_clean_data()
    
    # Validate processing
    validation = processor.validate_processing()
    
    print(f"\n🎯 Enhanced data processing completed successfully!")
    print(f"📈 Ready for machine learning model training with preserved achievement data!")


if __name__ == "__main__":
    main()
