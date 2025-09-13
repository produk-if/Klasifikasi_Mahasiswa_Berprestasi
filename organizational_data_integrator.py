"""
Organizational Data Integration Script

This script integrates the generated organizational data with the existing
enhanced student dataset to create a comprehensive dataset for classification.

Author: AI Assistant  
Date: September 13, 2025
Purpose: Complete organizational data integration for student achievement classification
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import logging

class OrganizationalDataIntegrator:
    """
    Integrate organizational data with existing student achievement dataset.
    """
    
    def __init__(self, output_dir='integrated_data'):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('OrganizationalIntegrator')
        
    def load_data(self, enhanced_data_path, organizational_features_path):
        """
        Load both enhanced student data and organizational features.
        
        Parameters:
        -----------
        enhanced_data_path : str
            Path to enhanced student dataset
        organizational_features_path : str  
            Path to organizational features dataset
            
        Returns:
        --------
        tuple : (enhanced_df, organizational_df)
        """
        self.logger.info("Loading datasets...")
        
        # Load enhanced student data
        enhanced_df = pd.read_csv(enhanced_data_path)
        self.logger.info(f"Loaded enhanced data: {enhanced_df.shape[0]} students, {enhanced_df.shape[1]} features")
        
        # Load organizational features
        organizational_df = pd.read_csv(organizational_features_path)
        self.logger.info(f"Loaded organizational features: {organizational_df.shape[0]} students, {organizational_df.shape[1]} features")
        
        return enhanced_df, organizational_df
        
    def integrate_datasets(self, enhanced_df, organizational_df):
        """
        Integrate organizational features with enhanced student data.
        
        Parameters:
        -----------
        enhanced_df : pandas.DataFrame
            Enhanced student dataset
        organizational_df : pandas.DataFrame
            Organizational features dataset
            
        Returns:
        --------
        pandas.DataFrame : Integrated dataset
        """
        self.logger.info("Integrating datasets...")
        
        # Ensure NIM columns are compatible - convert both to strings without decimal
        enhanced_df['nim'] = enhanced_df['nim'].astype(str)
        organizational_df['nim'] = organizational_df['nim'].astype('int64').astype(str)
        
        # Remove duplicate organizational columns if they exist in enhanced_df
        org_columns_to_remove = [
            'total_organizations', 'leadership_positions', 'leadership_duration_months',
            'org_type_diversity', 'avg_involvement_duration', 'current_active_orgs',
            'academic_orgs', 'social_orgs', 'organizational_score'
        ]
        
        existing_org_cols = [col for col in org_columns_to_remove if col in enhanced_df.columns]
        if existing_org_cols:
            self.logger.info(f"Removing existing organizational columns: {existing_org_cols}")
            enhanced_df = enhanced_df.drop(columns=existing_org_cols)
            
        # Merge datasets on NIM
        integrated_df = pd.merge(enhanced_df, organizational_df, on='nim', how='left')
        
        # Fill missing organizational data with zeros for students without involvement
        org_feature_cols = [col for col in organizational_df.columns if col != 'nim']
        integrated_df[org_feature_cols] = integrated_df[org_feature_cols].fillna(0)
        
        self.logger.info(f"Integration completed: {integrated_df.shape[0]} students, {integrated_df.shape[1]} features")
        
        return integrated_df
        
    def recalculate_organizational_score(self, integrated_df):
        """
        Recalculate organizational score using the new comprehensive features.
        
        Parameters:
        -----------
        integrated_df : pandas.DataFrame
            Integrated dataset
            
        Returns:
        --------
        pandas.DataFrame : Dataset with updated organizational score
        """
        self.logger.info("Recalculating organizational scores...")
        
        # Normalized organizational involvement score (0-1)
        max_orgs = integrated_df['total_organizations'].max()
        if max_orgs > 0:
            org_involvement = integrated_df['total_organizations'] / max_orgs
        else:
            org_involvement = 0
            
        # Leadership experience score (0-1)
        max_leadership = integrated_df['leadership_positions'].max()
        if max_leadership > 0:
            leadership_score = integrated_df['leadership_positions'] / max_leadership
        else:
            leadership_score = 0
            
        # Diversity score (0-1)
        max_diversity = integrated_df['org_type_diversity'].max()
        if max_diversity > 0:
            diversity_score = integrated_df['org_type_diversity'] / max_diversity
        else:
            diversity_score = 0
            
        # Duration experience score (0-1)  
        max_avg_duration = integrated_df['avg_involvement_duration'].max()
        if max_avg_duration > 0:
            duration_score = np.clip(integrated_df['avg_involvement_duration'] / 12.0, 0, 1)  # Cap at 12 months
        else:
            duration_score = 0
            
        # Community impact score (0-1)
        max_impact = integrated_df['community_impact_projects'].max()
        if max_impact > 0:
            impact_score = integrated_df['community_impact_projects'] / max_impact
        else:
            impact_score = 0
            
        # Weighted organizational score
        integrated_df['organizational_score'] = (
            0.25 * org_involvement +      # 25% - number of organizations
            0.30 * leadership_score +     # 30% - leadership positions
            0.15 * diversity_score +      # 15% - organization type diversity
            0.15 * duration_score +       # 15% - involvement duration
            0.15 * impact_score           # 15% - community impact
        )
        
        self.logger.info("Organizational scores recalculated")
        return integrated_df
        
    def update_composite_scoring(self, integrated_df):
        """
        Update composite scoring with new organizational features.
        
        Parameters:
        -----------
        integrated_df : pandas.DataFrame
            Dataset with updated organizational scores
            
        Returns:
        --------
        pandas.DataFrame : Dataset with updated composite scoring
        """
        self.logger.info("Updating composite scoring...")
        
        # Update weighted components with consistent weights (40-35-25)
        integrated_df['academic_weighted'] = integrated_df['academic_score'] * 0.40
        integrated_df['achievement_weighted'] = integrated_df['achievement_score'] * 0.35
        integrated_df['organizational_weighted'] = integrated_df['organizational_score'] * 0.25
        
        # Recalculate composite score
        integrated_df['composite_score'] = (
            integrated_df['academic_weighted'] +
            integrated_df['achievement_weighted'] + 
            integrated_df['organizational_weighted']
        )
        
        self.logger.info("Composite scoring updated")
        return integrated_df
        
    def update_classification_labels(self, integrated_df):
        """
        Update classification labels based on new comprehensive features.
        
        Parameters:
        -----------
        integrated_df : pandas.DataFrame
            Dataset with updated composite scores
            
        Returns:
        --------
        pandas.DataFrame : Dataset with updated labels
        """
        self.logger.info("Updating classification labels...")
        
        # Update criteria based on comprehensive features
        integrated_df['academic_excellence'] = (
            (integrated_df['academic_score'] >= 0.75) |
            (integrated_df['final_ipk'] >= 3.50)
        ).astype(int)
        
        integrated_df['achievement_portfolio'] = (
            (integrated_df['total_prestasi'] >= 2) |
            (integrated_df['achievement_score'] >= 0.3) |
            (integrated_df['international_achievements'] > 0)
        ).astype(int)
        
        integrated_df['leadership_experience'] = (
            (integrated_df['leadership_positions'] >= 1) |
            (integrated_df['highest_leadership_level'] >= 1) |
            (integrated_df['student_government_orgs'] >= 1)
        ).astype(int)
        
        # Count criteria met
        integrated_df['criteria_met'] = (
            integrated_df['academic_excellence'] +
            integrated_df['achievement_portfolio'] + 
            integrated_df['leadership_experience']
        )
        
        # Update berprestasi label (meet at least 2 of 3 criteria)
        integrated_df['berprestasi'] = (integrated_df['criteria_met'] >= 2).astype(int)
        
        # Update performance tier
        conditions = [
            integrated_df['criteria_met'] >= 3,
            integrated_df['criteria_met'] >= 2,
        ]
        choices = ['exemplary', 'proficient']
        integrated_df['performance_tier'] = np.select(conditions, choices, default='developing')
        
        # Log results
        pos_count = integrated_df['berprestasi'].sum()
        total_count = len(integrated_df)
        self.logger.info(f"Updated labels - Positive: {pos_count}/{total_count} ({pos_count/total_count*100:.1f}%)")
        
        return integrated_df
        
    def generate_integration_report(self, integrated_df):
        """
        Generate integration report comparing before and after.
        
        Parameters:
        -----------
        integrated_df : pandas.DataFrame
            Final integrated dataset
            
        Returns:
        --------
        str : Integration report
        """
        report = []
        report.append("🔗 ORGANIZATIONAL DATA INTEGRATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        report.append("📊 DATASET OVERVIEW:")
        report.append("-" * 20)
        report.append(f"• Total Students: {len(integrated_df)}")
        report.append(f"• Total Features: {len(integrated_df.columns)}")
        report.append(f"• Students with Organizational Involvement: {len(integrated_df[integrated_df['total_organizations'] > 0])}")
        report.append("")
        
        report.append("🏛️ ORGANIZATIONAL FEATURES SUMMARY:")
        report.append("-" * 35)
        report.append(f"• Average Organizations per Student: {integrated_df['total_organizations'].mean():.2f}")
        report.append(f"• Students with Leadership Roles: {len(integrated_df[integrated_df['leadership_positions'] > 0])}")
        report.append(f"• Average Organization Type Diversity: {integrated_df['org_type_diversity'].mean():.2f}")
        report.append(f"• Mean Organizational Score: {integrated_df['organizational_score'].mean():.3f}")
        report.append("")
        
        report.append("📈 UPDATED CLASSIFICATION RESULTS:")
        report.append("-" * 35)
        
        # Performance tier distribution
        tier_counts = integrated_df['performance_tier'].value_counts()
        total = len(integrated_df)
        
        for tier, count in tier_counts.items():
            percentage = (count / total) * 100
            report.append(f"• {tier.title()}: {count} students ({percentage:.1f}%)")
            
        report.append("")
        
        # Criteria analysis
        report.append("🎯 MULTI-CRITERIA ANALYSIS:")
        report.append("-" * 30)
        report.append(f"• Academic Excellence: {integrated_df['academic_excellence'].sum()} students ({integrated_df['academic_excellence'].mean()*100:.1f}%)")
        report.append(f"• Achievement Portfolio: {integrated_df['achievement_portfolio'].sum()} students ({integrated_df['achievement_portfolio'].mean()*100:.1f}%)")
        report.append(f"• Leadership Experience: {integrated_df['leadership_experience'].sum()} students ({integrated_df['leadership_experience'].mean()*100:.1f}%)")
        report.append("")
        
        # Final classification
        report.append("🏆 FINAL CLASSIFICATION:")
        report.append("-" * 25)
        pos_rate = integrated_df['berprestasi'].mean()
        report.append(f"• Berprestasi Students: {integrated_df['berprestasi'].sum()}/{total} ({pos_rate*100:.1f}%)")
        report.append(f"• Average Composite Score: {integrated_df['composite_score'].mean():.3f}")
        report.append("")
        
        report.append("✅ INTEGRATION QUALITY INDICATORS:")
        report.append("-" * 35)
        report.append(f"• No Missing Values: {integrated_df.isnull().sum().sum() == 0}")
        report.append(f"• Balanced Classification: {70 <= pos_rate*100 <= 90}")
        report.append(f"• Comprehensive Features: {len(integrated_df.columns) >= 45}")
        report.append(f"• Realistic Organizational Involvement: {60 <= (integrated_df['total_organizations'] > 0).mean()*100 <= 90}")
        
        return '\n'.join(report)
        
    def export_integrated_data(self, integrated_df, metadata=None):
        """
        Export the final integrated dataset.
        
        Parameters:
        -----------
        integrated_df : pandas.DataFrame
            Final integrated dataset
        metadata : dict, optional
            Additional metadata
        """
        # Export main dataset
        output_path = f'{self.output_dir}/integrated_enhanced_dataset_{self.timestamp}.csv'
        integrated_df.to_csv(output_path, index=False)
        self.logger.info(f"Exported integrated dataset to: {output_path}")
        
        # Export metadata
        export_metadata = {
            'integration_timestamp': self.timestamp,
            'total_students': int(len(integrated_df)),
            'total_features': len(integrated_df.columns),
            'feature_categories': {
                'academic': 8,
                'achievement': 10, 
                'organizational': 16,
                'composite': 5,
                'metadata': 8,
                'labels': len([col for col in integrated_df.columns if col in ['berprestasi', 'performance_tier', 'criteria_met']])
            },
            'classification_distribution': {
                'exemplary': int(len(integrated_df[integrated_df['performance_tier'] == 'exemplary'])),
                'proficient': int(len(integrated_df[integrated_df['performance_tier'] == 'proficient'])), 
                'developing': int(len(integrated_df[integrated_df['performance_tier'] == 'developing']))
            },
            'positive_class_rate': float(integrated_df['berprestasi'].mean()),
            'organizational_coverage': float((integrated_df['total_organizations'] > 0).mean())
        }
        
        if metadata:
            export_metadata.update(metadata)
            
        metadata_path = f'{self.output_dir}/integration_metadata_{self.timestamp}.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(export_metadata, f, indent=2)
        self.logger.info(f"Exported metadata to: {metadata_path}")
        
        return output_path

def main():
    """Main execution function."""
    print("🔗 Organizational Data Integration")
    print("=" * 40)
    
    # Configuration
    enhanced_data_path = "enhanced_clean_data/combined_enhanced_20250913_144318.csv"
    organizational_features_path = "organizational_data/organizational_features_20250913_150429.csv"
    output_dir = "integrated_data"
    
    # Check files exist
    if not os.path.exists(enhanced_data_path):
        print(f"❌ Error: Enhanced data file not found at {enhanced_data_path}")
        return
        
    if not os.path.exists(organizational_features_path):
        print(f"❌ Error: Organizational features file not found at {organizational_features_path}")
        return
    
    try:
        # Initialize integrator
        integrator = OrganizationalDataIntegrator(output_dir=output_dir)
        
        # Load data
        enhanced_df, organizational_df = integrator.load_data(
            enhanced_data_path, organizational_features_path
        )
        
        # Integrate datasets
        print("🔗 Integrating datasets...")
        integrated_df = integrator.integrate_datasets(enhanced_df, organizational_df)
        
        # Recalculate scores
        print("⚖️ Recalculating organizational scores...")
        integrated_df = integrator.recalculate_organizational_score(integrated_df)
        
        print("🎯 Updating composite scoring...")
        integrated_df = integrator.update_composite_scoring(integrated_df)
        
        print("🏷️ Updating classification labels...")
        integrated_df = integrator.update_classification_labels(integrated_df)
        
        # Generate report
        print("📋 Generating integration report...")
        report = integrator.generate_integration_report(integrated_df)
        print("\n" + report)
        
        # Export data
        print("💾 Exporting integrated dataset...")
        output_path = integrator.export_integrated_data(integrated_df)
        
        # Save report
        report_path = f"{output_dir}/integration_report_{integrator.timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Integration completed successfully!")
        print(f"📁 Integrated dataset: {output_path}")
        print(f"📊 Integration report: {report_path}")
        print(f"📂 All files available in: {output_dir}/")
        
    except Exception as e:
        print(f"❌ Integration failed: {str(e)}")
        logging.error(f"Integration failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
