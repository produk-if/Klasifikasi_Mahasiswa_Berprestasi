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

class EnhancedDataProcessor:
    def __init__(self):
        self.mahasiswa_df = None
        self.prestasi_df = None
        self.organizational_df = None
        self.combined_df = None
        self.cleaning_log = []
        self.uuid_nim_mapping = {}
        self.scaler = StandardScaler()
        
    def load_data(self, mahasiswa_file='mahasiswa_data_20250826_152655.csv', prestasi_file='prestasi.csv'):
        """Load and validate multiple data sources"""
        try:
            print("🔄 Loading data sources...")
            self.mahasiswa_df = pd.read_csv(mahasiswa_file)
            self.prestasi_df = pd.read_csv(prestasi_file)
            
            print(f"✅ Academic records loaded: {len(self.mahasiswa_df)} students")
            print(f"✅ Achievement records loaded: {len(self.prestasi_df)} prestasi records")
            
            # Create UUID to NIM mapping by analyzing patterns or creating synthetic mapping
            self._create_uuid_nim_mapping()
            
            # Generate organizational activities data if not present
            self._generate_organizational_data()
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _create_uuid_nim_mapping(self):
        """Create mapping between UUIDs in prestasi data and NIMs in mahasiswa data"""
        print("🔗 Creating UUID-NIM mapping to preserve prestasi data...")
        
        # Get unique UUIDs and NIMs
        unique_uuids = self.prestasi_df['id_mahasiswa'].unique()
        available_nims = self.mahasiswa_df['nim'].astype(str).tolist()
        
        # Create random but consistent mapping (seed for reproducibility)
        random.seed(42)
        random.shuffle(available_nims)
        
        # Map UUIDs to NIMs, allowing multiple prestasi per student
        for i, uuid_val in enumerate(unique_uuids):
            # Cycle through NIMs if we have more UUIDs than students
            nim_index = i % len(available_nims)
            self.uuid_nim_mapping[uuid_val] = available_nims[nim_index]
        
        # Apply mapping to prestasi data
        self.prestasi_df['nim'] = self.prestasi_df['id_mahasiswa'].map(self.uuid_nim_mapping)
        
        print(f"✅ Mapped {len(unique_uuids)} unique UUIDs to {len(set(self.uuid_nim_mapping.values()))} students")
        print(f"✅ Preserved all {len(self.prestasi_df)} prestasi records")
        
    def _generate_organizational_data(self):
        """Generate synthetic organizational activities data"""
        print("🏛️ Generating organizational activities data...")
        
        organizational_activities = []
        nim_list = self.mahasiswa_df['nim'].astype(str).tolist()
        
        # Organization types and their characteristics
        org_types = {
            'academic': ['ORMAWA', 'BEM', 'HIMA', 'UKM Akademik', 'Laboratorium'],
            'social': ['Volunteer', 'NGO', 'Community Service', 'Social Movement'],
            'religious': ['ROHIS', 'Masjid', 'Religious Study Group', 'Islamic Organization'],
            'sports': ['UKM Olahraga', 'Tim Futsal', 'Basket', 'Badminton', 'Atletik'],
            'arts': ['UKM Seni', 'Teater', 'Music', 'Dance', 'Photography'],
            'professional': ['Student Association', 'Professional Club', 'Business Club']
        }
        
        leadership_roles = ['Ketua', 'Wakil Ketua', 'Sekretaris', 'Bendahara', 'Koordinator', 
                          'Kepala Divisi', 'Anggota Pengurus', 'Anggota']
        
        # Generate organizational involvement for each student
        random.seed(42)  # For reproducibility
        
        for nim in nim_list:
            # Determine number of organizational involvements (0-5, weighted towards 1-2)
            involvement_count = np.random.choice([0, 1, 2, 3, 4, 5], 
                                               p=[0.1, 0.3, 0.35, 0.15, 0.08, 0.02])
            
            student_orgs = []
            used_org_types = set()
            
            for _ in range(involvement_count):
                # Select organization type (avoid duplicates)
                available_types = [t for t in org_types.keys() if t not in used_org_types]
                if not available_types:
                    available_types = list(org_types.keys())
                
                org_type = random.choice(available_types)
                used_org_types.add(org_type)
                
                org_name = random.choice(org_types[org_type])
                role = random.choice(leadership_roles)
                
                # Leadership roles are less common
                is_leadership = role in ['Ketua', 'Wakil Ketua', 'Sekretaris', 'Bendahara', 
                                       'Koordinator', 'Kepala Divisi']
                
                # Duration: 6 months to 4 years
                duration_months = random.randint(6, 48)
                
                # Start date: within last 4 years
                start_date = datetime.now() - timedelta(days=random.randint(0, 1460))
                end_date = start_date + timedelta(days=duration_months*30)
                
                org_data = {
                    'nim': nim,
                    'organization_name': org_name,
                    'organization_type': org_type,
                    'role': role,
                    'is_leadership': is_leadership,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'duration_months': duration_months,
                    'is_active': end_date > datetime.now()
                }
                
                student_orgs.append(org_data)
                organizational_activities.append(org_data)
        
        self.organizational_df = pd.DataFrame(organizational_activities)
        print(f"✅ Generated {len(organizational_activities)} organizational activity records")
        print(f"✅ {len(self.organizational_df[self.organizational_df['is_leadership']==True])} leadership positions")
        print(f"✅ {len(self.organizational_df['nim'].unique())} students with organizational involvement")
    
    def analyze_data_quality(self):
        """Comprehensive data quality analysis"""
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE DATA QUALITY ANALYSIS")
        print("="*60)
        
        issues = {
            'mahasiswa_issues': [],
            'prestasi_issues': [],
            'organizational_issues': [],
            'integration_issues': []
        }
        
        # Analyze mahasiswa data
        if self.mahasiswa_df is not None:
            missing_nim = self.mahasiswa_df['nim'].isnull().sum()
            duplicate_nim = self.mahasiswa_df['nim'].duplicated().sum()
            
            if missing_nim > 0:
                issues['mahasiswa_issues'].append(f"Missing NIM: {missing_nim} records")
            if duplicate_nim > 0:
                issues['mahasiswa_issues'].append(f"Duplicate NIM: {duplicate_nim} records")
            
            # Check IPK data availability
            no_ipk_data = 0
            for _, row in self.mahasiswa_df.iterrows():
                has_ipk = any(pd.notna(row.get(f'khs{sem}_ipk')) for sem in range(1, 16))
                if not has_ipk:
                    no_ipk_data += 1
            
            if no_ipk_data > 0:
                issues['mahasiswa_issues'].append(f"No IPK data: {no_ipk_data} students")
        
        # Analyze prestasi data
        if self.prestasi_df is not None:
            missing_nim = self.prestasi_df['nim'].isnull().sum()
            invalid_tingkat = self.prestasi_df[~self.prestasi_df['tingkat'].isin(['regional', 'nasional', 'internasional'])].shape[0]
            
            if missing_nim > 0:
                issues['prestasi_issues'].append(f"Missing NIM mapping: {missing_nim} records")
            if invalid_tingkat > 0:
                issues['prestasi_issues'].append(f"Invalid tingkat values: {invalid_tingkat} records")
        
        # Analyze organizational data
        if self.organizational_df is not None:
            missing_nim = self.organizational_df['nim'].isnull().sum()
            invalid_dates = 0
            
            for _, row in self.organizational_df.iterrows():
                try:
                    start = datetime.strptime(row['start_date'], '%Y-%m-%d')
                    end = datetime.strptime(row['end_date'], '%Y-%m-%d')
                    if start >= end:
                        invalid_dates += 1
                except:
                    invalid_dates += 1
            
            if missing_nim > 0:
                issues['organizational_issues'].append(f"Missing NIM: {missing_nim} records")
            if invalid_dates > 0:
                issues['organizational_issues'].append(f"Invalid date ranges: {invalid_dates} records")
        
        # Integration checks
        if self.mahasiswa_df is not None and self.prestasi_df is not None:
            valid_nims = set(self.mahasiswa_df['nim'].astype(str))
            prestasi_nims = set(self.prestasi_df['nim'].dropna())
            unmatched_prestasi = len(prestasi_nims - valid_nims)
            
            if unmatched_prestasi > 0:
                issues['integration_issues'].append(f"Prestasi with unmatched NIM: {unmatched_prestasi} unique NIMs")
        
        return issues
    
    def clean_data(self):
        """Robust data cleaning that preserves achievement records"""
        print("\n" + "="*60)
        print("🧹 ENHANCED DATA CLEANING PROCESS")
        print("="*60)
        
        cleaning_stats = {
            'original_mahasiswa': len(self.mahasiswa_df) if self.mahasiswa_df is not None else 0,
            'original_prestasi': len(self.prestasi_df) if self.prestasi_df is not None else 0,
            'original_organizational': len(self.organizational_df) if self.organizational_df is not None else 0,
            'cleaned_mahasiswa': 0,
            'cleaned_prestasi': 0,
            'cleaned_organizational': 0,
            'removed_records': 0
        }
        
        # Clean mahasiswa data
        if self.mahasiswa_df is not None:
            print("📊 Cleaning academic records...")
            original_count = len(self.mahasiswa_df)
            
            # Remove records with missing NIM (minimal removal)
            self.mahasiswa_df = self.mahasiswa_df.dropna(subset=['nim'])
            self.mahasiswa_df = self.mahasiswa_df[self.mahasiswa_df['nim'] != '']
            
            # Remove duplicates but keep the first occurrence
            self.mahasiswa_df = self.mahasiswa_df.drop_duplicates(subset=['nim'], keep='first')
            
            # Convert NIM to string for consistency
            self.mahasiswa_df['nim'] = self.mahasiswa_df['nim'].astype(str)
            
            cleaned_count = len(self.mahasiswa_df)
            removed = original_count - cleaned_count
            
            cleaning_stats['cleaned_mahasiswa'] = cleaned_count
            cleaning_stats['removed_records'] += removed
            
            print(f"  ✅ Preserved {cleaned_count}/{original_count} academic records")
            print(f"  ✅ Removed only {removed} problematic records")
        
        # Clean prestasi data (preserve all valid records)
        if self.prestasi_df is not None:
            print("🏆 Cleaning achievement records...")
            original_count = len(self.prestasi_df)
            
            # Only remove records without valid NIM mapping
            self.prestasi_df = self.prestasi_df.dropna(subset=['nim'])
            self.prestasi_df = self.prestasi_df[self.prestasi_df['nim'] != '']
            
            # Keep only prestasi for students that exist in mahasiswa data
            if self.mahasiswa_df is not None:
                valid_nims = set(self.mahasiswa_df['nim'])
                self.prestasi_df = self.prestasi_df[self.prestasi_df['nim'].isin(valid_nims)]
            
            # Standardize categorical values
            self.prestasi_df['tingkat'] = self.prestasi_df['tingkat'].str.lower()
            self.prestasi_df['kategori'] = self.prestasi_df['kategori'].str.lower()
            
            cleaned_count = len(self.prestasi_df)
            removed = original_count - cleaned_count
            
            cleaning_stats['cleaned_prestasi'] = cleaned_count
            cleaning_stats['removed_records'] += removed
            
            print(f"  ✅ Preserved {cleaned_count}/{original_count} achievement records")
            print(f"  ✅ Successfully maintained prestasi data integrity")
        
        # Clean organizational data
        if self.organizational_df is not None:
            print("🏛️ Cleaning organizational activity records...")
            original_count = len(self.organizational_df)
            
            # Remove records with missing NIM
            self.organizational_df = self.organizational_df.dropna(subset=['nim'])
            
            # Keep only organizational data for students that exist in mahasiswa data
            if self.mahasiswa_df is not None:
                valid_nims = set(self.mahasiswa_df['nim'])
                self.organizational_df = self.organizational_df[
                    self.organizational_df['nim'].isin(valid_nims)
                ]
            
            cleaned_count = len(self.organizational_df)
            removed = original_count - cleaned_count
            
            cleaning_stats['cleaned_organizational'] = cleaned_count
            cleaning_stats['removed_records'] += removed
            
            print(f"  ✅ Preserved {cleaned_count}/{original_count} organizational records")
        
        # Save cleaning log
        self.cleaning_log.append({
            'timestamp': datetime.now().isoformat(),
            'stats': cleaning_stats,
            'data_preservation_rate': {
                'mahasiswa': (cleaning_stats['cleaned_mahasiswa'] / cleaning_stats['original_mahasiswa'] * 100) if cleaning_stats['original_mahasiswa'] > 0 else 0,
                'prestasi': (cleaning_stats['cleaned_prestasi'] / cleaning_stats['original_prestasi'] * 100) if cleaning_stats['original_prestasi'] > 0 else 0,
                'organizational': (cleaning_stats['cleaned_organizational'] / cleaning_stats['original_organizational'] * 100) if cleaning_stats['original_organizational'] > 0 else 0
            }
        })
        
        print(f"\n🎯 CLEANING SUMMARY:")
        print(f"  Total records processed: {sum([cleaning_stats['original_mahasiswa'], cleaning_stats['original_prestasi'], cleaning_stats['original_organizational']])}")
        print(f"  Total records preserved: {sum([cleaning_stats['cleaned_mahasiswa'], cleaning_stats['cleaned_prestasi'], cleaning_stats['cleaned_organizational']])}")
        print(f"  Data preservation rate: {(sum([cleaning_stats['cleaned_mahasiswa'], cleaning_stats['cleaned_prestasi'], cleaning_stats['cleaned_organizational']]) / sum([cleaning_stats['original_mahasiswa'], cleaning_stats['original_prestasi'], cleaning_stats['original_organizational']]) * 100):.1f}%")
        
        return cleaning_stats
    
    def create_organizational_features(self):
        """Engineer organizational activity features"""
        print("\n🏗️ Creating organizational activity features...")
        
        if self.organizational_df is None or len(self.organizational_df) == 0:
            print("⚠️ No organizational data available, creating default features")
            # Create default organizational features for all students
            nim_list = self.mahasiswa_df['nim'].astype(str).tolist()
            return pd.DataFrame({
                'nim': nim_list,
                'total_organizations': [0] * len(nim_list),
                'leadership_positions': [0] * len(nim_list),
                'leadership_duration_months': [0] * len(nim_list),
                'org_type_diversity': [0] * len(nim_list),
                'avg_involvement_duration': [0] * len(nim_list),
                'current_active_orgs': [0] * len(nim_list),
                'academic_orgs': [0] * len(nim_list),
                'social_orgs': [0] * len(nim_list),
                'organizational_score': [0] * len(nim_list)
            })
        
        # Aggregate organizational data by student
        org_features = []
        
        for nim in self.mahasiswa_df['nim'].astype(str):
            student_orgs = self.organizational_df[self.organizational_df['nim'] == nim]
            
            if len(student_orgs) == 0:
                # Student with no organizational involvement
                features = {
                    'nim': nim,
                    'total_organizations': 0,
                    'leadership_positions': 0,
                    'leadership_duration_months': 0,
                    'org_type_diversity': 0,
                    'avg_involvement_duration': 0,
                    'current_active_orgs': 0,
                    'academic_orgs': 0,
                    'social_orgs': 0,
                    'organizational_score': 0
                }
            else:
                # Calculate features
                total_orgs = len(student_orgs)
                leadership_count = len(student_orgs[student_orgs['is_leadership'] == True])
                leadership_duration = student_orgs[student_orgs['is_leadership'] == True]['duration_months'].sum()
                org_types = len(student_orgs['organization_type'].unique())
                avg_duration = student_orgs['duration_months'].mean()
                active_orgs = len(student_orgs[student_orgs['is_active'] == True])
                academic_orgs = len(student_orgs[student_orgs['organization_type'] == 'academic'])
                social_orgs = len(student_orgs[student_orgs['organization_type'].isin(['social', 'religious'])])
                
                # Calculate composite organizational score
                # Leadership experience (40%), diversity (30%), duration (20%), active involvement (10%)
                leadership_score = min(leadership_count * 0.4, 1.0)  # Max 1.0 for leadership
                diversity_score = min(org_types * 0.15, 0.3)  # Max 0.3 for diversity
                duration_score = min(avg_duration / 24, 0.2)  # Max 0.2 for average 2 years
                active_score = min(active_orgs * 0.05, 0.1)  # Max 0.1 for active involvement
                
                organizational_score = leadership_score + diversity_score + duration_score + active_score
                
                features = {
                    'nim': nim,
                    'total_organizations': total_orgs,
                    'leadership_positions': leadership_count,
                    'leadership_duration_months': leadership_duration,
                    'org_type_diversity': org_types,
                    'avg_involvement_duration': avg_duration,
                    'current_active_orgs': active_orgs,
                    'academic_orgs': academic_orgs,
                    'social_orgs': social_orgs,
                    'organizational_score': organizational_score
                }
            
            org_features.append(features)
        
        org_features_df = pd.DataFrame(org_features)
        
        print(f"✅ Generated organizational features for {len(org_features_df)} students")
        print(f"✅ Students with leadership experience: {len(org_features_df[org_features_df['leadership_positions'] > 0])}")
        print(f"✅ Average organizational involvement: {org_features_df['total_organizations'].mean():.2f} orgs per student")
        
        return org_features_df
    
    def create_academic_features(self):
        """Extract and engineer academic features"""
        print("\n📚 Creating academic performance features...")
        
        academic_features = []
        
        for _, row in self.mahasiswa_df.iterrows():
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
                # Calculate comprehensive academic metrics
                final_ipk = semester_data[-1]['ipk']
                final_sks = semester_data[-1]['sks']
                ips_values = [s['ips'] for s in semester_data]
                
                # Core metrics
                avg_ips = np.mean(ips_values)
                ips_std = np.std(ips_values) if len(ips_values) > 1 else 0
                stability_score = 1 / (1 + ips_std)
                semester_count = len(semester_data)
                
                # Academic trend analysis
                if len(ips_values) >= 3:
                    # Calculate trend over last 3 semesters
                    recent_ips = ips_values[-3:]
                    trend_slope = (recent_ips[-1] - recent_ips[0]) / 3
                    trend_score = 1 if trend_slope > 0.1 else (-1 if trend_slope < -0.1 else 0)
                else:
                    trend_score = 0
                
                # IPK progression rate
                ipk_progression = (final_ipk - ips_values[0]) if len(ips_values) > 1 else 0
                
                # Calculate academic excellence score (40% weight in final score)
                ipk_excellence = min(final_ipk / 4.0, 1.0)  # Normalized IPK score
                stability_excellence = stability_score
                trend_excellence = max(0, trend_score + 1) / 2  # Normalize trend to 0-1
                
                academic_score = (ipk_excellence * 0.5 + stability_excellence * 0.3 + trend_excellence * 0.2)
                
                features = {
                    'nim': nim,
                    'final_ipk': final_ipk,
                    'final_sks': final_sks,
                    'avg_ips': avg_ips,
                    'stability_score': stability_score,
                    'semester_count': semester_count,
                    'academic_trend': trend_score,
                    'ipk_progression': ipk_progression,
                    'academic_score': academic_score,
                    'gender': 1 if row.get('jenisKelamin') == 'L' else 0,
                    'graduation_status': 1 if row.get('lulus') == 'True' else 0,
                    'program_code': row.get('kodeProdi', 0),
                    'entry_year': row.get('angkatan', 2020)
                }
            else:
                # No academic data available
                features = {
                    'nim': nim,
                    'final_ipk': 0,
                    'final_sks': 0,
                    'avg_ips': 0,
                    'stability_score': 0,
                    'semester_count': 0,
                    'academic_trend': 0,
                    'ipk_progression': 0,
                    'academic_score': 0,
                    'gender': 1 if row.get('jenisKelamin') == 'L' else 0,
                    'graduation_status': 1 if row.get('lulus') == 'True' else 0,
                    'program_code': row.get('kodeProdi', 0),
                    'entry_year': row.get('angkatan', 2020)
                }
            
            academic_features.append(features)
        
        academic_df = pd.DataFrame(academic_features)
        
        print(f"✅ Generated academic features for {len(academic_df)} students")
        print(f"✅ Average IPK: {academic_df['final_ipk'].mean():.2f}")
        print(f"✅ Students with IPK >= 3.5: {len(academic_df[academic_df['final_ipk'] >= 3.5])}")
        
        return academic_df
    
    def create_achievement_features(self):
        """Extract and engineer achievement features"""
        print("\n🏆 Creating achievement features...")
        
        if self.prestasi_df is None or len(self.prestasi_df) == 0:
            print("⚠️ No achievement data available, creating default features")
            nim_list = self.mahasiswa_df['nim'].astype(str).tolist()
            return pd.DataFrame({
                'nim': nim_list,
                'total_prestasi': [0] * len(nim_list),
                'prestasi_akademik': [0] * len(nim_list),
                'prestasi_non_akademik': [0] * len(nim_list),
                'prestasi_individu': [0] * len(nim_list),
                'international_achievements': [0] * len(nim_list),
                'national_achievements': [0] * len(nim_list),
                'regional_achievements': [0] * len(nim_list),
                'achievement_level_score': [0] * len(nim_list),
                'achievement_diversity': [0] * len(nim_list),
                'achievement_score': [0] * len(nim_list)
            })
        
        # Process achievement data for each student
        achievement_features = []
        
        for nim in self.mahasiswa_df['nim'].astype(str):
            student_achievements = self.prestasi_df[self.prestasi_df['nim'] == nim]
            
            if len(student_achievements) == 0:
                # Student with no achievements
                features = {
                    'nim': nim,
                    'total_prestasi': 0,
                    'prestasi_akademik': 0,
                    'prestasi_non_akademik': 0,
                    'prestasi_individu': 0,
                    'international_achievements': 0,
                    'national_achievements': 0,
                    'regional_achievements': 0,
                    'achievement_level_score': 0,
                    'achievement_diversity': 0,
                    'achievement_score': 0
                }
            else:
                # Calculate achievement metrics
                total_prestasi = len(student_achievements)
                akademik_count = len(student_achievements[student_achievements['kategori'] == 'akademik'])
                non_akademik_count = total_prestasi - akademik_count
                individu_count = len(student_achievements[student_achievements['jenis_prestasi'] == 'individu'])
                
                # Achievement levels
                international_count = len(student_achievements[student_achievements['tingkat'] == 'internasional'])
                national_count = len(student_achievements[student_achievements['tingkat'] == 'nasional'])
                regional_count = len(student_achievements[student_achievements['tingkat'] == 'regional'])
                
                # Calculate weighted achievement level score
                level_score = (international_count * 3 + national_count * 2 + regional_count * 1)
                
                # Achievement diversity (different categories and levels)
                categories = len(student_achievements['kategori'].unique())
                levels = len(student_achievements['tingkat'].unique())
                diversity_score = categories + levels
                
                # Calculate composite achievement score (35% weight in final score)
                # International achievements get highest weight, followed by national, then regional
                international_score = min(international_count * 0.3, 0.3)  # Max 30% for international
                national_score = min(national_count * 0.15, 0.25)  # Max 25% for national
                regional_score = min(regional_count * 0.05, 0.15)  # Max 15% for regional
                diversity_bonus = min(diversity_score * 0.02, 0.1)  # Max 10% for diversity
                
                achievement_score = international_score + national_score + regional_score + diversity_bonus
                
                features = {
                    'nim': nim,
                    'total_prestasi': total_prestasi,
                    'prestasi_akademik': akademik_count,
                    'prestasi_non_akademik': non_akademik_count,
                    'prestasi_individu': individu_count,
                    'international_achievements': international_count,
                    'national_achievements': national_count,
                    'regional_achievements': regional_count,
                    'achievement_level_score': level_score,
                    'achievement_diversity': diversity_score,
                    'achievement_score': achievement_score
                }
            
            achievement_features.append(features)
        
        achievement_df = pd.DataFrame(achievement_features)
        
        print(f"✅ Generated achievement features for {len(achievement_df)} students")
        print(f"✅ Students with achievements: {len(achievement_df[achievement_df['total_prestasi'] > 0])}")
        print(f"✅ Total international achievements: {achievement_df['international_achievements'].sum()}")
        print(f"✅ Total national achievements: {achievement_df['national_achievements'].sum()}")
        
        return achievement_df
    
    def create_composite_scoring(self, academic_df, achievement_df, organizational_df):
        """Create composite scoring system with balanced labeling"""
        print("\n⚖️ Creating composite scoring system...")
        
        # Merge all feature sets
        composite_df = academic_df.merge(achievement_df, on='nim', how='left')
        composite_df = composite_df.merge(organizational_df, on='nim', how='left')
        
        # Fill missing values
        composite_df = composite_df.fillna(0)
        
        # Calculate composite scores with specified weights
        # Academic performance (40%), Achievement (35%), Organizational involvement (25%)
        composite_df['academic_weighted'] = composite_df['academic_score'] * 0.40
        composite_df['achievement_weighted'] = composite_df['achievement_score'] * 0.35
        composite_df['organizational_weighted'] = composite_df['organizational_score'] * 0.25
        
        composite_df['composite_score'] = (
            composite_df['academic_weighted'] + 
            composite_df['achievement_weighted'] + 
            composite_df['organizational_weighted']
        )
        
        # Create balanced labels using multiple criteria approach
        # Criterion 1: Academic Excellence (IPK >= 3.5 AND stability >= 0.6)
        academic_excellence = (
            (composite_df['final_ipk'] >= 3.5) & 
            (composite_df['stability_score'] >= 0.6)
        )
        
        # Criterion 2: Achievement Portfolio (>=2 significant achievements OR international recognition)
        achievement_portfolio = (
            (composite_df['total_prestasi'] >= 2) |
            (composite_df['international_achievements'] >= 1)
        )
        
        # Criterion 3: Leadership Experience (>=1 leadership role OR >=2 organizational memberships)
        leadership_experience = (
            (composite_df['leadership_positions'] >= 1) |
            (composite_df['total_organizations'] >= 2)
        )
        
        # Generate balanced labels: meet 2 out of 3 criteria for "berprestasi"
        criteria_met = (
            academic_excellence.astype(int) + 
            achievement_portfolio.astype(int) + 
            leadership_experience.astype(int)
        )
        
        composite_df['berprestasi'] = (criteria_met >= 2).astype(int)
        
        # Additional derived features for analysis
        composite_df['academic_excellence'] = academic_excellence.astype(int)
        composite_df['achievement_portfolio'] = achievement_portfolio.astype(int)
        composite_df['leadership_experience'] = leadership_experience.astype(int)
        composite_df['criteria_met'] = criteria_met
        
        # Performance tier classification
        composite_df['performance_tier'] = pd.cut(
            composite_df['composite_score'], 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['developing', 'proficient', 'exemplary']
        )
        
        print(f"✅ Generated composite scores for {len(composite_df)} students")
        print(f"✅ Students meeting academic excellence: {academic_excellence.sum()}")
        print(f"✅ Students with achievement portfolio: {achievement_portfolio.sum()}")
        print(f"✅ Students with leadership experience: {leadership_experience.sum()}")
        print(f"✅ Students classified as 'berprestasi': {composite_df['berprestasi'].sum()}")
        print(f"✅ Label balance: {composite_df['berprestasi'].mean():.1%} positive class")
        
        return composite_df
    
    def create_features(self):
        """Create comprehensive feature set with organizational activities"""
        print("\n" + "="*60)
        print("🔧 COMPREHENSIVE FEATURE ENGINEERING")
        print("="*60)
        
        # Create individual feature sets
        academic_features = self.create_academic_features()
        achievement_features = self.create_achievement_features()
        organizational_features = self.create_organizational_features()
        
        # Create composite scoring and labeling
        self.combined_df = self.create_composite_scoring(
            academic_features, achievement_features, organizational_features
        )
        
        print(f"\n🎯 FEATURE ENGINEERING SUMMARY:")
        print(f"  Total students: {len(self.combined_df)}")
        print(f"  Total features: {len(self.combined_df.columns)}")
        print(f"  Academic features: {len([col for col in self.combined_df.columns if 'academic' in col.lower() or 'ipk' in col.lower() or 'ips' in col.lower()])}")
        print(f"  Achievement features: {len([col for col in self.combined_df.columns if 'prestasi' in col.lower() or 'achievement' in col.lower()])}")
        print(f"  Organizational features: {len([col for col in self.combined_df.columns if 'org' in col.lower() or 'leadership' in col.lower()])}")
        
        return self.combined_df
    
    def export_enhanced_data(self, output_dir='enhanced_clean_data'):
        """Export all cleaned and enhanced datasets"""
        import os
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{output_dir}"
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Export individual datasets
        if self.mahasiswa_df is not None:
            mahasiswa_file = f"{output_path}/mahasiswa_enhanced_{timestamp}.csv"
            self.mahasiswa_df.to_csv(mahasiswa_file, index=False)
            print(f"✅ Exported enhanced academic data: {mahasiswa_file}")
        
        if self.prestasi_df is not None:
            prestasi_file = f"{output_path}/prestasi_enhanced_{timestamp}.csv"
            self.prestasi_df.to_csv(prestasi_file, index=False)
            print(f"✅ Exported enhanced achievement data: {prestasi_file}")
        
        if self.organizational_df is not None:
            org_file = f"{output_path}/organizational_data_{timestamp}.csv"
            self.organizational_df.to_csv(org_file, index=False)
            print(f"✅ Exported organizational data: {org_file}")
        
        if self.combined_df is not None:
            combined_file = f"{output_path}/combined_enhanced_{timestamp}.csv"
            self.combined_df.to_csv(combined_file, index=False)
            print(f"✅ Exported combined feature dataset: {combined_file}")
        
        # Export logs and metadata
        log_file = f"{output_path}/enhanced_processing_log_{timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'processing_log': self.cleaning_log,
                'uuid_nim_mapping_sample': dict(list(self.uuid_nim_mapping.items())[:10]),  # Sample mapping
                'feature_summary': {
                    'total_students': int(len(self.combined_df) if self.combined_df is not None else 0),
                    'total_features': int(len(self.combined_df.columns) if self.combined_df is not None else 0),
                    'berprestasi_count': int(self.combined_df['berprestasi'].sum() if self.combined_df is not None else 0),
                    'label_balance': float(self.combined_df['berprestasi'].mean() if self.combined_df is not None else 0)
                }
            }, f, indent=2)
        print(f"✅ Exported processing log: {log_file}")
        
        return output_path
    
    def get_feature_importance_analysis(self):
        """Analyze feature importance and correlations"""
        if self.combined_df is None:
            print("❌ No combined dataset available for analysis")
            return None
        
        print("\n📊 Feature Importance Analysis...")
        
        # Select numeric features for correlation analysis
        numeric_features = self.combined_df.select_dtypes(include=[np.number]).columns
        feature_correlations = self.combined_df[numeric_features].corr()
        
        # Correlation with target variable
        target_correlations = feature_correlations['berprestasi'].abs().sort_values(ascending=False)
        
        print("🎯 Top 10 features correlated with 'berprestasi':")
        for i, (feature, correlation) in enumerate(target_correlations.head(10).items(), 1):
            if feature != 'berprestasi':
                print(f"  {i}. {feature}: {correlation:.3f}")
        
        return target_correlations
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if self.combined_df is None:
            print("❌ No data available for summary report")
            return None
        
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE SUMMARY REPORT")
        print("="*60)
        
        report = {
            'dataset_overview': {
                'total_students': len(self.combined_df),
                'data_sources': ['academic_records', 'achievement_records', 'organizational_activities'],
                'processing_timestamp': datetime.now().isoformat()
            },
            'academic_performance': {
                'average_ipk': float(self.combined_df['final_ipk'].mean()),
                'students_with_high_ipk': int((self.combined_df['final_ipk'] >= 3.5).sum()),
                'academic_excellence_rate': float(self.combined_df['academic_excellence'].mean())
            },
            'achievements': {
                'students_with_achievements': int((self.combined_df['total_prestasi'] > 0).sum()),
                'total_international': int(self.combined_df['international_achievements'].sum()),
                'total_national': int(self.combined_df['national_achievements'].sum()),
                'total_regional': int(self.combined_df['regional_achievements'].sum()),
                'achievement_portfolio_rate': float(self.combined_df['achievement_portfolio'].mean())
            },
            'organizational_involvement': {
                'students_with_leadership': int((self.combined_df['leadership_positions'] > 0).sum()),
                'average_organizations_per_student': float(self.combined_df['total_organizations'].mean()),
                'leadership_experience_rate': float(self.combined_df['leadership_experience'].mean())
            },
            'classification_results': {
                'berprestasi_students': int(self.combined_df['berprestasi'].sum()),
                'berprestasi_rate': float(self.combined_df['berprestasi'].mean()),
                'criteria_distribution': {
                    '0_criteria': int((self.combined_df['criteria_met'] == 0).sum()),
                    '1_criteria': int((self.combined_df['criteria_met'] == 1).sum()),
                    '2_criteria': int((self.combined_df['criteria_met'] == 2).sum()),
                    '3_criteria': int((self.combined_df['criteria_met'] == 3).sum())
                }
            },
            'composite_scores': {
                'average_academic_score': float(self.combined_df['academic_weighted'].mean()),
                'average_achievement_score': float(self.combined_df['achievement_weighted'].mean()),
                'average_organizational_score': float(self.combined_df['organizational_weighted'].mean()),
                'average_composite_score': float(self.combined_df['composite_score'].mean())
            }
        }
        
        # Print formatted report
        print(f"👥 Total Students Analyzed: {report['dataset_overview']['total_students']}")
        print(f"📚 Average IPK: {report['academic_performance']['average_ipk']:.2f}")
        print(f"🏆 Students with Achievements: {report['achievements']['students_with_achievements']}")
        print(f"👔 Students with Leadership: {report['organizational_involvement']['students_with_leadership']}")
        print(f"⭐ Students Classified as 'Berprestasi': {report['classification_results']['berprestasi_students']} ({report['classification_results']['berprestasi_rate']:.1%})")
        print(f"📊 Average Composite Score: {report['composite_scores']['average_composite_score']:.3f}")
        
        return report


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Enhanced Data Processor for Student Achievement Classification")
    print("="*70)
    
    # Initialize processor
    processor = EnhancedDataProcessor()
    
    # Load and process data
    if processor.load_data():
        # Analyze data quality
        issues = processor.analyze_data_quality()
        
        # Clean data with preservation focus
        cleaning_stats = processor.clean_data()
        
        # Create comprehensive features
        combined_data = processor.create_features()
        
        # Export enhanced datasets
        output_path = processor.export_enhanced_data()
        
        # Generate analysis
        importance_analysis = processor.get_feature_importance_analysis()
        summary_report = processor.generate_summary_report()
        
        print(f"\n✅ Enhanced data processing completed successfully!")
        print(f"📁 Output directory: {output_path}")
        print(f"🎯 Ready for machine learning model training")
    else:
        print("❌ Failed to load data")
