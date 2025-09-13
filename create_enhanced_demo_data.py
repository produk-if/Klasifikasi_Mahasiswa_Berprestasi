import pandas as pd
import numpy as np
import uuid
import json
import os
from datetime import datetime, timedelta

def create_enhanced_demo_data():
    """
    Create comprehensive demo data that demonstrates the enhanced data processor
    with proper ID mapping between academic records and achievements
    """
    
    np.random.seed(42)  # For reproducible results
    
    print("🎯 Creating Enhanced Demo Data with Proper ID Mapping")
    print("="*70)
    
    # Step 1: Load existing academic data as base
    try:
        mahasiswa_df = pd.read_csv('mahasiswa_data_20250826_152655.csv')
        print(f"✅ Loaded {len(mahasiswa_df)} academic records as base")
    except:
        print("❌ Could not load existing academic data")
        return
    
    # Step 2: Create student mapping with both NIM and UUID
    student_mapping = []
    for _, student in mahasiswa_df.iterrows():
        student_mapping.append({
            'nim': student['nim'],
            'student_uuid': str(uuid.uuid4()),
            'nama': student.get('nama', f'Student_{student["nim"]}'),
            'angkatan': student['angkatan'],
            'prodi': student.get('kodeProdi', 'Unknown')
        })
    
    mapping_df = pd.DataFrame(student_mapping)
    print(f"✅ Created student mapping with UUIDs for {len(mapping_df)} students")
    
    # Step 3: Create realistic achievement data with proper mapping
    achievements_data = []
    achievement_types = {
        'akademik': [
            'Lomba Karya Tulis Ilmiah', 'Essay Competition', 'Research Paper Competition',
            'Scientific Paper Competition', 'Academic Writing Contest', 'Thesis Competition',
            'Journal Publication', 'Conference Presentation', 'Academic Excellence Award'
        ],
        'non_akademik': [
            'Programming Contest', 'Hackathon', 'Design Competition', 'Innovation Challenge',
            'Entrepreneurship Competition', 'Leadership Award', 'Community Service Award',
            'Sports Championship', 'Arts Competition', 'Cultural Performance'
        ]
    }
    
    levels = ['lokal', 'regional', 'nasional', 'internasional']
    level_weights = [0.4, 0.35, 0.2, 0.05]  # Most achievements at local/regional level
    
    # 60% of students have achievements (realistic percentage)
    students_with_achievements = mapping_df.sample(frac=0.6, random_state=42)
    
    for _, student in students_with_achievements.iterrows():
        # Number of achievements per student (1-5, weighted towards fewer)
        num_achievements = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.2, 0.08, 0.02])
        
        for i in range(num_achievements):
            # Choose category
            kategori = np.random.choice(['akademik', 'non_akademik'], p=[0.4, 0.6])
            judul = np.random.choice(achievement_types[kategori])
            
            # Choose level (higher achievers more likely to have higher levels)
            tingkat = np.random.choice(levels, p=level_weights)
            
            # Create achievement record
            achievement = {
                'id': str(uuid.uuid4()),
                'judul': judul,
                'tingkat': tingkat,
                'kategori': kategori,
                'tanggal': (datetime.now() - timedelta(days=np.random.randint(30, 1000))).strftime('%d/%m/%Y'),
                'id_mahasiswa': student['student_uuid'],  # Use UUID from mapping
                'nim': student['nim'],  # Include NIM for verification
                'nama': student['nama'],
                'dibuat_pada': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                'jenis_prestasi': 'individu',
                'posisi': np.random.choice(['Juara 1', 'Juara 2', 'Juara 3', 'Finalis', 'Peserta Terbaik'], 
                                         p=[0.2, 0.2, 0.2, 0.25, 0.15])
            }
            
            achievements_data.append(achievement)
    
    achievements_df = pd.DataFrame(achievements_data)
    print(f"✅ Created {len(achievements_df)} achievement records for {len(students_with_achievements)} students")
    
    # Step 4: Create organizational activities data
    org_data = []
    org_types = {
        'academic': ['BEM Fakultas', 'Himpunan Mahasiswa Prodi', 'Unit Kegiatan Mahasiswa', 'ORMAWA'],
        'social': ['Karang Taruna', 'Komunitas Peduli Sosial', 'Relawan Bencana', 'Tim Bakti Sosial'],
        'religious': ['Kerohanian Islam', 'Persekutuan Kristen', 'Dharma Wacana', 'Kegiatan Keagamaan'],
        'sports': ['Unit Olahraga', 'Tim Futsal', 'Basket Club', 'Badminton Club'],
        'arts': ['Unit Kesenian', 'Teater Kampus', 'Paduan Suara', 'Tari Tradisional'],
        'technology': ['IT Club', 'Robotika', 'Programming Community', 'Tech Startup'],
        'volunteer': ['PMI', 'Pramuka', 'KKN', 'Volunteer Community']
    }
    
    # 75% of students have organizational involvement
    students_with_orgs = mapping_df.sample(frac=0.75, random_state=43)
    
    for _, student in students_with_orgs.iterrows():
        # Number of organizations (1-3, most students in 1-2)
        num_orgs = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        
        selected_types = np.random.choice(list(org_types.keys()), size=min(num_orgs, len(org_types)), replace=False)
        
        for org_type in selected_types:
            org_name = np.random.choice(org_types[org_type])
            
            # Leadership probability based on student profile
            is_leader = np.random.random() < 0.25  # 25% leadership rate
            
            if is_leader:
                role = np.random.choice(['Ketua', 'Wakil Ketua', 'Sekretaris', 'Bendahara', 'Koordinator'])
            else:
                role = 'Anggota'
            
            # Duration in semesters (1-8)
            duration = np.random.randint(1, 9)
            
            # Start year relative to student's entrance year
            start_year = int(student['angkatan']) + np.random.randint(0, 2)
            
            org_record = {
                'nim': student['nim'],
                'student_uuid': student['student_uuid'],
                'organization_name': org_name,
                'organization_type': org_type,
                'role': role,
                'is_leadership': is_leader,
                'start_year': start_year,
                'duration_semesters': duration,
                'activity_level': np.random.choice(['active', 'very_active', 'moderately_active'], 
                                                 p=[0.5, 0.3, 0.2])
            }
            
            org_data.append(org_record)
    
    org_df = pd.DataFrame(org_data)
    print(f"✅ Created {len(org_df)} organizational records for {len(students_with_orgs)} students")
    
    # Step 5: Export demo data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save student mapping
    mapping_df.to_csv(f'demo_student_mapping_{timestamp}.csv', index=False)
    
    # Save achievements with proper ID mapping
    achievements_df.to_csv(f'demo_prestasi_mapped_{timestamp}.csv', index=False)
    
    # Save organizational data
    org_df.to_csv(f'demo_organizational_{timestamp}.csv', index=False)
    
    # Create summary statistics
    summary = {
        'creation_timestamp': timestamp,
        'total_students': len(mapping_df),
        'students_with_achievements': len(students_with_achievements),
        'total_achievements': len(achievements_df),
        'students_with_organizations': len(students_with_orgs),
        'total_organizational_records': len(org_df),
        'achievement_distribution': {
            'akademik': len(achievements_df[achievements_df['kategori'] == 'akademik']),
            'non_akademik': len(achievements_df[achievements_df['kategori'] == 'non_akademik'])
        },
        'level_distribution': achievements_df['tingkat'].value_counts().to_dict(),
        'organization_distribution': org_df['organization_type'].value_counts().to_dict(),
        'leadership_positions': int(org_df['is_leadership'].sum())
    }
    
    # Save summary
    with open(f'demo_data_summary_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📊 Demo Data Summary:")
    print(f"   - Total students: {summary['total_students']}")
    print(f"   - Students with achievements: {summary['students_with_achievements']} ({summary['students_with_achievements']/summary['total_students']*100:.1f}%)")
    print(f"   - Total achievements: {summary['total_achievements']}")
    print(f"   - Students with organizational involvement: {summary['students_with_organizations']} ({summary['students_with_organizations']/summary['total_students']*100:.1f}%)")
    print(f"   - Leadership positions: {summary['leadership_positions']}")
    
    print(f"\n📁 Files created:")
    print(f"   - demo_student_mapping_{timestamp}.csv")
    print(f"   - demo_prestasi_mapped_{timestamp}.csv")
    print(f"   - demo_organizational_{timestamp}.csv")
    print(f"   - demo_data_summary_{timestamp}.json")
    
    return {
        'mapping_df': mapping_df,
        'achievements_df': achievements_df,
        'org_df': org_df,
        'summary': summary,
        'timestamp': timestamp
    }

if __name__ == "__main__":
    result = create_enhanced_demo_data()
    print(f"\n🎉 Enhanced demo data creation completed successfully!")
    print(f"🔄 You can now test the enhanced_data_processor.py with properly mapped demo data")
