import re

# Step 1: Normalize and clean address
def clean_address(addr):
    if pd.isna(addr):
        return ""
    addr = addr.upper()
    addr = re.sub(r'[^A-Z0-9,./\- ]', '', addr)  # Remove special characters except basic punctuations
    addr = re.sub(r'\s+', ' ', addr)  # Remove extra whitespace
    return addr.strip()

df['ALAMAT_CLEAN'] = df['ALAMAT'].apply(clean_address)

# Step 2: Create a rough dictionary to infer city/province based on known keywords
location_dict = {
    'JAKARTA': ('JAKARTA', 'DKI JAKARTA'),
    'BEKASI': ('BEKASI', 'JAWA BARAT'),
    'BOGOR': ('BOGOR', 'JAWA BARAT'),
    'TANGERANG': ('TANGERANG', 'BANTEN'),
    'BANDUNG': ('BANDUNG', 'JAWA BARAT'),
    'SUKABUMI': ('SUKABUMI', 'JAWA BARAT'),
    'DEPOK': ('DEPOK', 'JAWA BARAT'),
    'PEKANBARU': ('PEKANBARU', 'RIAU'),
    'MEDAN': ('MEDAN', 'SUMATERA UTARA'),
    'SURABAYA': ('SURABAYA', 'JAWA TIMUR'),
    'SEMARANG': ('SEMARANG', 'JAWA TENGAH'),
    'YOGYAKARTA': ('YOGYAKARTA', 'DI YOGYAKARTA'),
    'SOLO': ('SOLO', 'JAWA TENGAH'),
    'MALANG': ('MALANG', 'JAWA TIMUR'),
    'BALI': ('DENPASAR', 'BALI'),
    'BALIKPAPAN': ('BALIKPAPAN', 'KALIMANTAN TIMUR'),
    'SAMARINDA': ('SAMARINDA', 'KALIMANTAN TIMUR'),
    'MAKASSAR': ('MAKASSAR', 'SULAWESI SELATAN')
}

# Step 3: Guess the city and province
def guess_location(address):
    for keyword in location_dict:
        if keyword in address:
            return location_dict[keyword]
    return (None, None)

df[['ESTIMATED_CITY', 'ESTIMATED_PROVINCE']] = df['ALAMAT_CLEAN'].apply(
    lambda x: pd.Series(guess_location(x))
)

import ace_tools as tools; tools.display_dataframe_to_user(name="Cleaned Customer Address Data", dataframe=df)


