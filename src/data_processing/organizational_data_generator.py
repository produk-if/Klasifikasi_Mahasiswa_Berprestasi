"""
Organizational Data Generator for Indonesian University Context

This module generates realistic synthetic data for student organizational activities
that matches Indonesian university patterns and structures.

Author: AI Assistant
Date: September 13, 2025
Purpose: Student Achievement Classification Research - Organizational Data Simulation
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class OrganizationInfo:
    """Data class for organization information."""
    name: str
    type: str
    category: str
    selectivity: float  # How selective the org is (0-1)
    time_commitment: str  # Low, Medium, High
    leadership_levels: List[str]
    typical_duration: int  # Months

class IndonesianOrganizationalDataGenerator:
    """
    Generate realistic organizational activities data for Indonesian university students.
    """
    
    def __init__(self, output_dir='organizational_data', random_state=42):
        """
        Initialize the organizational data generator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save generated data
        random_state : int
            Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.random_state = random_state
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize organization data
        self._initialize_organizations()
        
        # Student involvement patterns
        self.involvement_rates = {
            'none': 0.15,           # 15% no involvement
            'low': 0.35,            # 35% minimal involvement
            'moderate': 0.35,       # 35% moderate involvement  
            'high': 0.15            # 15% high involvement
        }
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.output_dir}/generation_log_{self.timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('OrganizationalDataGenerator')
        
    def _initialize_organizations(self):
        """Initialize the database of university organizations."""
        
        self.organizations = {
            # Student Government Organizations
            'student_government': [
                OrganizationInfo("BEM Fakultas", "student_government", "governance", 0.8, "High", 
                               ["Anggota", "Staff Ahli", "Wakil Ketua", "Ketua"], 12),
                OrganizationInfo("DPM Fakultas", "student_government", "governance", 0.7, "Medium", 
                               ["Anggota", "Wakil Ketua", "Ketua"], 12),
                OrganizationInfo("Senat Mahasiswa", "student_government", "governance", 0.9, "High", 
                               ["Senator", "Wakil Ketua", "Ketua"], 12),
                OrganizationInfo("ORMAWA Jurusan", "student_government", "governance", 0.6, "Medium", 
                               ["Anggota", "Koordinator", "Ketua"], 12)
            ],
            
            # Academic Organizations
            'academic': [
                OrganizationInfo("Kelompok Studi Penelitian", "academic", "research", 0.6, "Medium", 
                               ["Anggota", "Asisten Peneliti", "Koordinator"], 6),
                OrganizationInfo("Tim Olimpiade", "academic", "competition", 0.8, "High", 
                               ["Anggota", "Koordinator Tim", "Ketua Tim"], 4),
                OrganizationInfo("Laboratorium Riset", "academic", "research", 0.7, "High", 
                               ["Asisten", "Koordinator", "Kepala Lab"], 12),
                OrganizationInfo("Tutor Sebaya", "academic", "education", 0.5, "Low", 
                               ["Tutor", "Koordinator", "Supervisor"], 6),
                OrganizationInfo("Journal Club", "academic", "research", 0.4, "Low", 
                               ["Anggota", "Presenter", "Koordinator"], 12)
            ],
            
            # Religious Organizations
            'religious': [
                OrganizationInfo("Kerohanian Islam", "religious", "spiritual", 0.3, "Medium", 
                               ["Anggota", "Koordinator Divisi", "Ketua"], 12),
                OrganizationInfo("Persekutuan Doa Kristen", "religious", "spiritual", 0.3, "Medium", 
                               ["Anggota", "Koordinator", "Ketua"], 12),
                OrganizationInfo("Dharma Wacana Buddha", "religious", "spiritual", 0.4, "Low", 
                               ["Anggota", "Koordinator"], 12),
                OrganizationInfo("Keluarga Mahasiswa Hindu", "religious", "spiritual", 0.4, "Low", 
                               ["Anggota", "Koordinator"], 12)
            ],
            
            # Social Service Organizations
            'social_service': [
                OrganizationInfo("KKN Tematik", "social_service", "community", 0.5, "Medium", 
                               ["Peserta", "Koordinator Desa", "Koordinator Program"], 2),
                OrganizationInfo("Relawan Bencana", "social_service", "emergency", 0.6, "High", 
                               ["Relawan", "Koordinator Tim", "Koordinator Wilayah"], 12),
                OrganizationInfo("Bakti Sosial Mahasiswa", "social_service", "community", 0.3, "Low", 
                               ["Relawan", "Koordinator Kegiatan", "Ketua"], 12),
                OrganizationInfo("Pendampingan Masyarakat", "social_service", "community", 0.7, "Medium", 
                               ["Pendamping", "Koordinator Lapangan", "Project Manager"], 6),
                OrganizationInfo("Donor Darah", "social_service", "health", 0.2, "Low", 
                               ["Donor", "Koordinator Event", "Ketua"], 12)
            ],
            
            # Special Interest Clubs
            'special_interest': [
                OrganizationInfo("Tim Debat", "special_interest", "competition", 0.8, "High", 
                               ["Anggota", "Koordinator Training", "Ketua Tim"], 12),
                OrganizationInfo("UKM Olahraga", "special_interest", "sports", 0.4, "Medium", 
                               ["Anggota", "Koordinator Cabang", "Ketua"], 12),
                OrganizationInfo("Paduan Suara", "special_interest", "arts", 0.6, "Medium", 
                               ["Anggota", "Section Leader", "Konduktor"], 12),
                OrganizationInfo("Fotografi", "special_interest", "arts", 0.3, "Low", 
                               ["Anggota", "Koordinator Event", "Ketua"], 12),
                OrganizationInfo("Robotika", "special_interest", "technology", 0.7, "High", 
                               ["Anggota", "Team Leader", "Project Manager"], 12),
                OrganizationInfo("Teater", "special_interest", "arts", 0.5, "Medium", 
                               ["Pemain", "Asisten Sutradara", "Sutradara"], 6),
                OrganizationInfo("Jurnalistik", "special_interest", "media", 0.4, "Medium", 
                               ["Reporter", "Editor", "Pemimpin Redaksi"], 12)
            ],
            
            # Professional Associations
            'professional': [
                OrganizationInfo("Himpunan Mahasiswa Teknik", "professional", "engineering", 0.5, "Medium", 
                               ["Anggota", "Koordinator Divisi", "Ketua"], 12),
                OrganizationInfo("Ikatan Mahasiswa Akuntansi", "professional", "business", 0.4, "Medium", 
                               ["Anggota", "Wakil Ketua", "Ketua"], 12),
                OrganizationInfo("Asosiasi Mahasiswa Psikologi", "professional", "psychology", 0.6, "Medium", 
                               ["Anggota", "Koordinator Program", "Ketua"], 12),
                OrganizationInfo("Forum Mahasiswa Kedokteran", "professional", "medical", 0.7, "High", 
                               ["Anggota", "Koordinator Riset", "Ketua"], 12),
                OrganizationInfo("Perhimpunan Mahasiswa Hukum", "professional", "law", 0.5, "Medium", 
                               ["Anggota", "Koordinator Kajian", "Ketua"], 12)
            ]
        }
        
        # Flatten organization list for easier access
        self.all_organizations = []
        for org_type, orgs in self.organizations.items():
            for org in orgs:
                org.type = org_type  # Ensure type is set correctly
                self.all_organizations.append(org)
                
        self.logger.info(f"Initialized {len(self.all_organizations)} organizations across {len(self.organizations)} categories")
        
    def _determine_involvement_level(self, academic_score):
        """
        Determine student involvement level based on academic performance and random factors.
        
        Parameters:
        -----------
        academic_score : float
            Student's academic performance score
            
        Returns:
        --------
        str : Involvement level ('none', 'low', 'moderate', 'high')
        """
        # Base probabilities
        base_probs = list(self.involvement_rates.values())
        
        # Adjust probabilities based on academic score
        if academic_score > 0.85:  # High achievers
            # More likely to have moderate to high involvement
            adjusted_probs = [0.10, 0.25, 0.45, 0.20]
        elif academic_score > 0.70:  # Good performers
            # Standard distribution
            adjusted_probs = [0.12, 0.33, 0.40, 0.15]
        elif academic_score > 0.50:  # Average performers
            # More likely to have low to moderate involvement
            adjusted_probs = [0.20, 0.45, 0.30, 0.05]
        else:  # Low performers
            # More likely to have no or low involvement
            adjusted_probs = [0.30, 0.50, 0.18, 0.02]
            
        levels = list(self.involvement_rates.keys())
        return np.random.choice(levels, p=adjusted_probs)
        
    def _select_organizations_for_student(self, involvement_level, student_profile):
        """
        Select appropriate organizations for a student based on involvement level and profile.
        
        Parameters:
        -----------
        involvement_level : str
            Student's involvement level
        student_profile : dict
            Student's profile information
            
        Returns:
        --------
        list : Selected organizations
        """
        if involvement_level == 'none':
            return []
            
        # Determine number of organizations
        org_counts = {
            'low': (1, 2),      # 1-2 organizations
            'moderate': (2, 4),  # 2-4 organizations
            'high': (3, 6)       # 3-6 organizations
        }
        
        min_orgs, max_orgs = org_counts[involvement_level]
        num_orgs = np.random.randint(min_orgs, max_orgs + 1)
        
        # Select organizations with realistic preferences
        selected_orgs = []
        available_orgs = self.all_organizations.copy()
        
        # First organization - based on student interests and academic level
        if student_profile['academic_score'] > 0.8:
            # High achievers prefer academic/professional orgs
            preferred_types = ['academic', 'professional', 'student_government']
        elif student_profile['academic_score'] > 0.6:
            # Good performers balance academic and interest
            preferred_types = ['academic', 'professional', 'special_interest', 'social_service']
        else:
            # Others prefer social/interest activities
            preferred_types = ['special_interest', 'social_service', 'religious']
            
        # Filter by preferred types for first selection
        preferred_orgs = [org for org in available_orgs if org.type in preferred_types]
        if preferred_orgs:
            first_org = np.random.choice(preferred_orgs)
            selected_orgs.append(first_org)
            available_orgs.remove(first_org)
            
        # Select remaining organizations
        for _ in range(num_orgs - len(selected_orgs)):
            if not available_orgs:
                break
                
            # Apply selectivity filter
            eligible_orgs = []
            for org in available_orgs:
                selection_prob = (1 - org.selectivity) + (org.selectivity * student_profile['academic_score'])
                if np.random.random() < selection_prob:
                    eligible_orgs.append(org)
                    
            if eligible_orgs:
                selected_org = np.random.choice(eligible_orgs)
                selected_orgs.append(selected_org)
                available_orgs.remove(selected_org)
                
        return selected_orgs
        
    def _generate_role_progression(self, org, involvement_duration):
        """
        Generate realistic role progression within an organization.
        
        Parameters:
        -----------
        org : OrganizationInfo
            Organization information
        involvement_duration : int
            Duration of involvement in months
            
        Returns:
        --------
        list : Role progression history
        """
        roles = org.leadership_levels
        progression = []
        
        # Start with basic role
        current_role = roles[0]
        months_in_role = 0
        
        for month in range(involvement_duration):
            months_in_role += 1
            
            # Check for promotion opportunity
            if len(roles) > 1 and current_role != roles[-1]:
                current_role_index = roles.index(current_role)
                
                # Promotion probability increases with time and org selectivity
                base_promotion_prob = 0.02  # 2% per month
                promotion_prob = base_promotion_prob * (1 + org.selectivity) * (months_in_role / 6)
                
                if np.random.random() < promotion_prob:
                    current_role = roles[current_role_index + 1]
                    months_in_role = 0
                    
            progression.append((month + 1, current_role))
            
        return progression
        
    def _calculate_organizational_features(self, org_history):
        """
        Calculate organizational involvement features from history.
        
        Parameters:
        -----------
        org_history : list
            List of organizational involvements
            
        Returns:
        --------
        dict : Organizational features
        """
        if not org_history:
            return {
                'total_organizations': 0,
                'leadership_positions': 0,
                'leadership_duration_months': 0,
                'org_type_diversity': 0,
                'avg_involvement_duration': 0,
                'current_active_orgs': 0,
                'academic_orgs': 0,
                'social_orgs': 0,
                'professional_orgs': 0,
                'student_government_orgs': 0,
                'religious_orgs': 0,
                'special_interest_orgs': 0,
                'highest_leadership_level': 0,
                'event_organization_experience': 0,
                'inter_org_collaboration': 0,
                'community_impact_projects': 0
            }
            
        # Basic counts
        total_orgs = len(org_history)
        org_types = set()
        type_counts = {
            'academic': 0,
            'social_service': 0,
            'professional': 0,
            'student_government': 0,
            'religious': 0,
            'special_interest': 0
        }
        
        total_duration = 0
        leadership_positions = 0
        leadership_duration = 0
        highest_level = 0
        current_active = 0
        
        for involvement in org_history:
            org_type = involvement['organization_type']
            duration = involvement['duration_months']
            roles = involvement['roles']
            is_current = involvement['is_current']
            
            org_types.add(org_type)
            if org_type in type_counts:
                type_counts[org_type] += 1
            
            total_duration += duration
            if is_current:
                current_active += 1
                
            # Analyze roles
            for role_info in roles:
                role = role_info['role']
                if role != involvement['organization_info'].leadership_levels[0]:  # Not base member role
                    leadership_positions += 1
                    leadership_duration += duration
                    
                # Calculate leadership level (0 = member, higher = more senior)
                try:
                    level = involvement['organization_info'].leadership_levels.index(role)
                    highest_level = max(highest_level, level)
                except ValueError:
                    pass
                    
        # Calculate derived features
        avg_duration = total_duration / total_orgs if total_orgs > 0 else 0
        org_diversity = len(org_types)
        
        # Estimate additional features based on involvement
        event_experience = min(total_orgs * 2, 10)  # Cap at 10
        collaboration_score = min(org_diversity * 0.3, 1.0)
        impact_projects = min(type_counts['social_service'] * 2 + type_counts['student_government'], 5)
        
        return {
            'total_organizations': total_orgs,
            'leadership_positions': leadership_positions,
            'leadership_duration_months': leadership_duration,
            'org_type_diversity': org_diversity,
            'avg_involvement_duration': avg_duration,
            'current_active_orgs': current_active,
            'academic_orgs': type_counts['academic'],
            'social_orgs': type_counts['social_service'],
            'professional_orgs': type_counts['professional'],
            'student_government_orgs': type_counts['student_government'],
            'religious_orgs': type_counts['religious'],
            'special_interest_orgs': type_counts['special_interest'],
            'highest_leadership_level': highest_level,
            'event_organization_experience': event_experience,
            'inter_org_collaboration': collaboration_score,
            'community_impact_projects': impact_projects
        }
        
    def generate_student_organizational_data(self, students_df):
        """
        Generate organizational data for all students.
        
        Parameters:
        -----------
        students_df : pandas.DataFrame
            DataFrame containing student information
            
        Returns:
        --------
        pandas.DataFrame : Organizational data
        """
        self.logger.info(f"Generating organizational data for {len(students_df)} students...")
        
        organizational_data = []
        
        for idx, student in students_df.iterrows():
            nim = student['nim']
            
            # Create student profile
            student_profile = {
                'academic_score': student.get('academic_score', 0.7),
                'program': student.get('program_code', '00000'),
                'entry_year': student.get('entry_year', 2020)
            }
            
            # Determine involvement level
            involvement_level = self._determine_involvement_level(student_profile['academic_score'])
            
            # Select organizations
            selected_orgs = self._select_organizations_for_student(involvement_level, student_profile)
            
            # Generate involvement history
            org_history = []
            for org in selected_orgs:
                # Determine involvement duration
                base_duration = org.typical_duration
                duration_variation = np.random.normal(0, base_duration * 0.3)
                actual_duration = max(1, int(base_duration + duration_variation))
                
                # Determine if currently active (70% chance)
                is_current = np.random.random() < 0.7
                
                # Generate role progression
                role_progression = self._generate_role_progression(org, actual_duration)
                
                involvement = {
                    'nim': nim,
                    'organization_name': org.name,
                    'organization_type': org.type,
                    'organization_category': org.category,
                    'organization_info': org,
                    'duration_months': actual_duration,
                    'is_current': is_current,
                    'roles': [{'month': month, 'role': role} for month, role in role_progression]
                }
                
                org_history.append(involvement)
                organizational_data.append({
                    'nim': nim,
                    'organization_name': org.name,
                    'organization_type': org.type,
                    'organization_category': org.category,
                    'duration_months': actual_duration,
                    'is_current_member': is_current,
                    'highest_role': role_progression[-1][1] if role_progression else org.leadership_levels[0],
                    'roles_held': len(set(role for _, role in role_progression)),
                    'selectivity_score': org.selectivity,
                    'time_commitment': org.time_commitment
                })
            
            # Calculate features for this student
            features = self._calculate_organizational_features(org_history)
            
            # Store individual student features (for integration with main dataset)
            student_features = {'nim': nim}
            student_features.update(features)
            
            if idx == 0 or idx % 20 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(students_df)} students - {involvement_level} involvement")
                
        self.logger.info(f"Generated {len(organizational_data)} organizational involvements")
        
        return pd.DataFrame(organizational_data)
        
    def generate_feature_dataset(self, students_df):
        """
        Generate organizational features dataset for integration.
        
        Parameters:
        -----------
        students_df : pandas.DataFrame
            DataFrame containing student information
            
        Returns:
        --------
        pandas.DataFrame : Features dataset
        """
        self.logger.info("Generating organizational features dataset...")
        
        features_data = []
        
        for idx, student in students_df.iterrows():
            nim = student['nim']
            
            # Create student profile
            student_profile = {
                'academic_score': student.get('academic_score', 0.7),
                'program': student.get('program_code', '00000'),
                'entry_year': student.get('entry_year', 2020)
            }
            
            # Determine involvement level
            involvement_level = self._determine_involvement_level(student_profile['academic_score'])
            
            # Select organizations
            selected_orgs = self._select_organizations_for_student(involvement_level, student_profile)
            
            # Generate involvement history
            org_history = []
            for org in selected_orgs:
                base_duration = org.typical_duration
                duration_variation = np.random.normal(0, base_duration * 0.3)
                actual_duration = max(1, int(base_duration + duration_variation))
                is_current = np.random.random() < 0.7
                role_progression = self._generate_role_progression(org, actual_duration)
                
                involvement = {
                    'organization_type': org.type,
                    'organization_info': org,
                    'duration_months': actual_duration,
                    'is_current': is_current,
                    'roles': [{'role': role} for _, role in role_progression]
                }
                org_history.append(involvement)
            
            # Calculate features
            features = self._calculate_organizational_features(org_history)
            features['nim'] = nim
            features_data.append(features)
            
        return pd.DataFrame(features_data)
        
    def export_data(self, organizational_df, features_df, metadata=None):
        """
        Export generated data to files.
        
        Parameters:
        -----------
        organizational_df : pandas.DataFrame
            Detailed organizational involvement data
        features_df : pandas.DataFrame
            Features dataset for integration
        metadata : dict, optional
            Additional metadata to include
        """
        # Export detailed data
        detailed_path = f'{self.output_dir}/organizational_involvements_{self.timestamp}.csv'
        organizational_df.to_csv(detailed_path, index=False)
        self.logger.info(f"Exported detailed organizational data to: {detailed_path}")
        
        # Export features data
        features_path = f'{self.output_dir}/organizational_features_{self.timestamp}.csv'
        features_df.to_csv(features_path, index=False)
        self.logger.info(f"Exported features data to: {features_path}")
        
        # Export metadata
        export_metadata = {
            'generation_timestamp': self.timestamp,
            'total_students': int(len(features_df)),
            'total_involvements': int(len(organizational_df)),
            'organization_types': list(self.organizations.keys()),
            'total_organizations': len(self.all_organizations),
            'involvement_distribution': {
                'none': int(len(features_df[features_df['total_organizations'] == 0])),
                'low': int(len(features_df[(features_df['total_organizations'] >= 1) & 
                                     (features_df['total_organizations'] <= 2)])),
                'moderate': int(len(features_df[(features_df['total_organizations'] >= 3) & 
                                          (features_df['total_organizations'] <= 4)])),
                'high': int(len(features_df[features_df['total_organizations'] >= 5]))
            },
            'leadership_statistics': {
                'students_with_leadership': int(len(features_df[features_df['leadership_positions'] > 0])),
                'average_leadership_duration': float(features_df['leadership_duration_months'].mean()),
                'max_organizations_per_student': int(features_df['total_organizations'].max())
            }
        }
        
        if metadata:
            export_metadata.update(metadata)
            
        metadata_path = f'{self.output_dir}/generation_metadata_{self.timestamp}.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(export_metadata, f, indent=2)
        self.logger.info(f"Exported metadata to: {metadata_path}")
        
    def generate_quality_report(self, features_df):
        """
        Generate a quality assessment report for the generated data.
        
        Parameters:
        -----------
        features_df : pandas.DataFrame
            Features dataset to analyze
            
        Returns:
        --------
        str : Quality report
        """
        report = []
        report.append("🏛️ ORGANIZATIONAL DATA QUALITY REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Basic statistics
        total_students = len(features_df)
        involved_students = len(features_df[features_df['total_organizations'] > 0])
        involvement_rate = (involved_students / total_students) * 100
        
        report.append("📊 INVOLVEMENT STATISTICS:")
        report.append("-" * 25)
        report.append(f"• Total Students: {total_students}")
        report.append(f"• Students with Involvement: {involved_students} ({involvement_rate:.1f}%)")
        report.append(f"• Average Organizations per Student: {features_df['total_organizations'].mean():.2f}")
        report.append(f"• Students with Leadership Roles: {len(features_df[features_df['leadership_positions'] > 0])}")
        report.append("")
        
        # Distribution analysis
        report.append("📈 INVOLVEMENT DISTRIBUTION:")
        report.append("-" * 30)
        for level, count in {
            'No Involvement (0)': len(features_df[features_df['total_organizations'] == 0]),
            'Low (1-2 orgs)': len(features_df[(features_df['total_organizations'] >= 1) & 
                                            (features_df['total_organizations'] <= 2)]),
            'Moderate (3-4 orgs)': len(features_df[(features_df['total_organizations'] >= 3) & 
                                                 (features_df['total_organizations'] <= 4)]),
            'High (5+ orgs)': len(features_df[features_df['total_organizations'] >= 5])
        }.items():
            percentage = (count / total_students) * 100
            report.append(f"• {level}: {count} students ({percentage:.1f}%)")
        
        report.append("")
        
        # Organization type distribution
        report.append("🏢 ORGANIZATION TYPE DISTRIBUTION:")
        report.append("-" * 35)
        org_type_cols = ['academic_orgs', 'social_orgs', 'professional_orgs', 
                        'student_government_orgs', 'religious_orgs', 'special_interest_orgs']
        
        for col in org_type_cols:
            if col in features_df.columns:
                count = (features_df[col] > 0).sum()
                percentage = (count / total_students) * 100
                org_type = col.replace('_orgs', '').replace('_', ' ').title()
                report.append(f"• {org_type}: {count} students ({percentage:.1f}%)")
        
        report.append("")
        
        # Leadership analysis
        report.append("👑 LEADERSHIP ANALYSIS:")
        report.append("-" * 25)
        leadership_students = features_df[features_df['leadership_positions'] > 0]
        if len(leadership_students) > 0:
            report.append(f"• Students with Leadership: {len(leadership_students)} ({len(leadership_students)/total_students*100:.1f}%)")
            report.append(f"• Average Leadership Duration: {leadership_students['leadership_duration_months'].mean():.1f} months")
            report.append(f"• Highest Leadership Level Achieved: {features_df['highest_leadership_level'].max()}")
        else:
            report.append("• No leadership positions recorded")
            
        report.append("")
        
        # Data quality indicators
        report.append("✅ DATA QUALITY INDICATORS:")
        report.append("-" * 30)
        report.append(f"• No Missing Values: {features_df.isnull().sum().sum() == 0}")
        report.append(f"• Realistic Involvement Rates: {60 <= involvement_rate <= 85}")
        report.append(f"• Diverse Organization Types: {features_df['org_type_diversity'].mean():.2f} avg types per student")
        report.append(f"• Reasonable Duration Values: {features_df['avg_involvement_duration'].mean():.1f} avg months")
        
        return '\n'.join(report)

def load_student_data(file_path):
    """Load student data from the enhanced dataset."""
    try:
        df = pd.read_csv(file_path)
        # Select relevant columns for organizational data generation
        student_cols = ['nim', 'academic_score', 'program_code', 'entry_year']
        available_cols = [col for col in student_cols if col in df.columns]
        
        if 'nim' not in available_cols:
            raise ValueError("NIM column is required for organizational data generation")
            
        return df[available_cols].copy()
    except Exception as e:
        logging.error(f"Error loading student data: {e}")
        raise

def main():
    """Main execution function."""
    print("🏛️ Indonesian University Organizational Data Generator")
    print("=" * 55)
    
    # Configuration
    student_data_path = "enhanced_clean_data/combined_enhanced_20250913_144318.csv"
    output_dir = "organizational_data"
    
    # Check if student data file exists
    if not os.path.exists(student_data_path):
        print(f"❌ Error: Student data file not found at {student_data_path}")
        print("Please ensure the enhanced dataset is available.")
        return
    
    try:
        # Load student data
        print(f"📂 Loading student data from: {student_data_path}")
        students_df = load_student_data(student_data_path)
        print(f"✅ Loaded data for {len(students_df)} students")
        
        # Initialize generator
        generator = IndonesianOrganizationalDataGenerator(output_dir=output_dir, random_state=42)
        
        # Generate organizational data
        print("🏗️ Generating organizational involvement data...")
        organizational_df = generator.generate_student_organizational_data(students_df)
        
        print("📊 Generating features dataset...")
        features_df = generator.generate_feature_dataset(students_df)
        
        # Export data
        print("💾 Exporting generated data...")
        generator.export_data(organizational_df, features_df)
        
        # Generate quality report
        print("📋 Generating quality report...")
        quality_report = generator.generate_quality_report(features_df)
        print("\n" + quality_report)
        
        # Save quality report
        report_path = f"{output_dir}/quality_report_{generator.timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(quality_report)
        
        print(f"\n✅ Organizational data generation completed successfully!")
        print(f"📁 Results available in: {output_dir}/")
        print(f"📊 Quality report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Generation failed: {str(e)}")
        logging.error(f"Generation failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
