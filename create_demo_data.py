import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime
import json

def create_demo_data():
    """Create demo data that matches between mahasiswa and prestasi"""
    print("🎯 CREATING DEMO DATA FOR DASHBOARD")
    print("="*60)
    
    # Load original mahasiswa data
    mahasiswa_df = pd.read_csv('mahasiswa_data_20250826_152655.csv')
    
    # Take first 50 students for demo
    demo_mahasiswa = mahasiswa_df.head(50).copy()
    
    # Create matching prestasi data
    prestasi_data = []
    categories = ['akademik', 'non_akademik']
    tingkat_levels = ['lokal', 'regional', 'nasional', 'internasional']
    jenis_prestasi = ['individu', 'tim']
    
    prestasi_titles = [
        'Lomba Karya Tulis Ilmiah', 'Kompetisi Programming', 'Hackathon',
        'Lomba Desain Grafis', 'Kompetisi Matematika', 'Olimpiade Sains',
        'Lomba Debat', 'Kompetisi Business Plan', 'Lomba Inovasi',
        'Festival Film Pendek', 'Lomba Fotografi', 'Kompetisi Robotika',
        'Lomba Esai', 'Turnamen E-Sports', 'Lomba Poster',
        'Kompetisi Aplikasi Mobile', 'Lomba Video Kreatif', 'Olimpiade Informatika'
    ]
    
    # Create prestasi for 60% of students (30 students)
    students_with_prestasi = random.sample(list(demo_mahasiswa['nim']), 30)
    
    for nim in students_with_prestasi:
        # Each student gets 1-5 prestasi
        num_prestasi = random.randint(1, 5)
        
        for _ in range(num_prestasi):
            prestasi_data.append({
                'id': str(uuid.uuid4()),
                'judul': random.choice(prestasi_titles),
                'tingkat': random.choice(tingkat_levels),
                'kategori': random.choice(categories),
                'tanggal': f"{random.randint(1, 12)}/{random.randint(1, 28)}/{random.randint(2020, 2024)} 00:00:00",
                'id_mahasiswa': str(nim),  # Use NIM directly
                'dibuat_pada': datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")[:-3],
                'jenis_prestasi': random.choice(jenis_prestasi),
                'institusi_tim': f"Tim {random.randint(1, 10)}",
                'nama_tim': f"Team_{random.randint(100, 999)}",
                'tahun': random.randint(2020, 2024)
            })
    
    prestasi_df = pd.DataFrame(prestasi_data)
    
    # Save demo data
    demo_mahasiswa.to_csv('demo_mahasiswa.csv', index=False)
    prestasi_df.to_csv('demo_prestasi.csv', index=False)
    
    print(f"✅ Demo mahasiswa data: {len(demo_mahasiswa)} records")
    print(f"✅ Demo prestasi data: {len(prestasi_df)} records") 
    print(f"✅ Students with prestasi: {len(students_with_prestasi)} out of {len(demo_mahasiswa)}")
    
    # Create data summary for dashboard
    summary = {
        'total_mahasiswa': len(demo_mahasiswa),
        'total_prestasi': len(prestasi_df),
        'mahasiswa_berprestasi': len(students_with_prestasi),
        'prestasi_by_tingkat': prestasi_df['tingkat'].value_counts().to_dict(),
        'prestasi_by_kategori': prestasi_df['kategori'].value_counts().to_dict(),
        'created_at': datetime.now().isoformat()
    }
    
    with open('demo_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Summary saved to demo_summary.json")
    return demo_mahasiswa, prestasi_df

if __name__ == "__main__":
    create_demo_data()
    print("\n🌐 Now you can use demo_mahasiswa.csv and demo_prestasi.csv in the dashboard!")
