"""
DRIAMS-A 2017 id file contains spectra codes with no preprocessed spectra. 
Remove overlap from id files.
"""

import pandas as pd


def remove_missing(id_2017):

    df_2017 = pd.read_csv(id_2017, low_memory=False)   
    df_2017 = df_2017.set_index('code') 
    print('shape df_2017 {}'.format(df_2017.shape))

    # codes in id file with no spectra present in DRIAMS-A/preprocessed/2017
    # determined through list_missing_sepctra.py from maldi-learn package
    missing_ids = ['6c2c74b6-a941-42fe-8bf4-a6f30b7b42bc_MALDI1',
                'af66e336-44f4-416d-898d-b70f33d32d88_MALDI1',
                '1f72183f-67c6-493a-973e-1a0c38a05bd3_MALDI1',
                '4177c04a-bbc6-4dbf-ba60-b1abc6d9e240_MALDI1',
                '6acae072-e3cf-41fc-95dd-ac3ce152c1ed_MALDI1',
                'd53a9431-dfd1-4a60-9ebb-c463b5aea38d_MALDI1',
                '049d01a7-8dcc-488f-babe-3caa67a7d4b8_MALDI1',
                'bdcbe1c3-e2ef-4fdb-8d15-53f95a0111b9_MALDI1',
                '51dce601-ff26-40a2-8068-4ac6c05df8fb_MALDI1',
                '06e91c76-8abf-4ecd-b3e2-b5c2ba095739_MALDI1',
                'b75f1f8f-c817-40b4-881b-fab84addf901_MALDI1',
                '835a08dd-4f75-403e-885d-e0d23c3ab846_MALDI1',
                '7d292b81-ea10-4c05-8b3c-a4047b602b3b_MALDI1',
                '9e98cd2d-c7e1-4d4b-810b-2a1a4b3c00cc_MALDI1',
                '01a97b6d-1307-4c9a-bcad-8b376616681d_MALDI1',
                '59535a92-fffa-4229-9e2f-1705a5f6e0c3_MALDI1',
                '20e6a8c0-d849-4379-ab1b-32347af01aeb_MALDI1',
                '3f5ce13c-03d3-4e05-a895-827ea2728dd3_MALDI1',
                '40adc757-30e3-45c6-8741-34e2880664ec_MALDI1',
                '4a2090ba-9910-4b95-b3dd-dfacc003e1d5_MALDI1',
                '58272f88-cd7b-4eed-b0a6-ae0509761c89_MALDI1',
                '714a2f71-b262-46e0-83bb-16232a09c4bf_MALDI1',
                '6f3cbd9d-1f64-467d-b5dd-7259d756c87f_MALDI1',
                'd6c75298-01cd-4e9b-bbf2-866752868650_MALDI1',
                '4abfaf98-057a-4373-be39-c08c688b82da_MALDI1',
                '6209e985-fbbc-4d9f-94d0-fcd944ed1a34_MALDI1',
                '40204ae3-926e-499e-aad0-7aa5a7d9ba8e_MALDI1',
                '00be1978-ffe6-4dd7-87b4-7230c316c968_MALDI1',
                'ebf6968b-0d3b-49a3-866b-7b34b51e9c98_MALDI1',
                '7d3c82a4-ddff-4e8e-8c5f-a7e2a92cc9db_MALDI1',
                '9cca748b-39ce-404b-9954-3119bc491b14_MALDI1',
                '7ea9d94f-3b79-4165-a9f0-a7904bda0cd3_MALDI1',
                'f44328dd-e0bd-422f-b51d-8d6ac5151c74_MALDI1',
                '48b8c51b-f8a9-47e0-a809-e08b636dc973_MALDI1',
                'c26d82ff-8c9d-49ed-8046-b95fc746db59_MALDI1',
                '185b4656-2758-44ae-8e38-20da34f3b5ef_MALDI1',
                '08aa6d25-e2c2-4bbb-b3f0-7c3e2fb3394b_MALDI1',
                '9861d1f3-3460-48f0-b527-92a38e9d118a_MALDI1',
                'fa429125-d64b-41d8-b308-75ace14ac741_MALDI1',
                '21b0c03c-1912-48d2-83c0-1e7901b30698_MALDI1',
                '2b9741eb-b93a-4ab8-a456-c0f4ec502f7a_MALDI1',
                '3e98e76c-06d6-4270-90c2-f97b499da620_MALDI1',
                '87eba357-a86d-49c0-84e5-d247a2c328ec_MALDI1',
                'b626ec35-c319-41dc-bddc-1739a53df4ab_MALDI1',
                'e5869845-ecee-4421-997c-6386955f0a40_MALDI1',
                'b0b3f78f-6a95-46de-98cb-2ebbdcc6f774_MALDI1',
                'dca96666-7915-4671-9aa5-be2642993d41_MALDI1',
                '3cf5c8bf-018b-4f04-a793-c96701dfb5e4_MALDI1',
                'c6123789-4e4b-4e0e-8796-72d4ddf9193a_MALDI1',
                '593a2932-af20-4c8c-b9f8-6907aa032c75_MALDI1',
                '68cf8bc2-b385-4553-bb83-d6f19349ac8a_MALDI1',
                '3a9ca2d5-a0f3-4d3f-a726-aed08771868d_MALDI1',
                'e4c98602-87af-4874-aea4-38337d42c398_MALDI1',
                '20db3b80-2ea3-45c9-a5c4-10fdc3a2e265_MALDI1',
                'e56e7754-023e-4bd7-8252-3b542f1161ec_MALDI1',
                '7dd3c001-79e0-4430-b5b8-5dc611f2de43_MALDI1',
                'c01732bb-108e-42e6-b840-45c39dfbe33c_MALDI1',
                '59726bbb-d364-4ce7-a45e-c2c97cfd3844_MALDI1',
                '1ae096d5-8f54-4a35-b545-7a9154aebeac_MALDI1',
                '288ae405-f9bc-4f72-b1c9-d14ad46199d2_MALDI1',
                'bc13723c-2b03-47a0-8841-9f0c4342127f_MALDI1',
                '82b222b9-92d8-4d50-8df5-a2b8e0f4daa7_MALDI1',
                'b29f6b8e-b00c-4f74-afe3-5b56e2180501_MALDI1',
                'ebb70154-9b8d-43ab-8781-2916de5121f4_MALDI1',
                '9005baad-126e-4d0d-9fba-d064166c5daa_MALDI1',
                '69390009-0cc8-4e98-aef3-180a3fa57a40_MALDI1',
                '103bb196-ef20-4a15-a20d-886af56626ad_MALDI1',
                '05b67602-e51c-4052-b857-81e8297cb44b_MALDI1',
                '2ca56ee2-1f72-4ec9-b9f6-ed260a6e5c12_MALDI1',
                '43ffd96f-f28d-42bc-9ec8-9ec5c7a510a2_MALDI1',
                'fefc4d37-451c-4722-bc61-8533754d6ff6_MALDI1',
                'e357a6d9-0b7b-4d62-a2f6-a260acdcc681_MALDI1',
                '85ad1add-c811-4ffe-8f53-a021e80f1006_MALDI1',
                '7e0e133a-ea45-41b0-9104-4fb3b295fdd9_MALDI1',
                '7ea28a1e-5d8a-4599-a3cf-33f88e101a26_MALDI1',
                '00c7a879-15ef-4eb5-9abe-289fcef6c6c3_MALDI1',
                '7f406f83-4c9a-4011-a639-a00e5cdc350d_MALDI1',
                'c8b7bdeb-b0f0-48fb-ba81-972d9d3fadd3_MALDI1',
                'dda6c49c-2b69-40d2-b552-8af3655b9d85_MALDI1',
                'bffa4e90-f6f3-4b7f-a85e-d0c214acf40e_MALDI1',
                'be40038d-035f-48bf-ae18-e2a2e10fe924_MALDI1',
                '55f6411a-b0e9-4f69-881d-9bca58d5f7fa_MALDI1',
                '2e07c7b3-55f5-4fb3-91f1-b5f192704465_MALDI1',
                'c6d7383f-5878-4b0d-bda9-d4496230eb87_MALDI1',
                '8045605f-4f96-4cd6-98d9-bf75b7b76629_MALDI1',
                '41031f89-d860-4fab-8731-45042dcb6ce1_MALDI1',
                   ]
    print(f'{len(missing_ids)} codes to be removed from 2017_strat.csv')

    # drop codes
    print(df_2017.head())
    [df_2017.drop(index=code, inplace=True) for code in missing_ids]

    # check sizes
    print('\nAfter removal:')
    print('shape df_2017 {}'.format(df_2017.shape))

    # write output to file
    df_2017.to_csv(id_2017)

if __name__ == '__main__':
    
    clean_id_2017 = './2017_strat.csv'
    remove_missing(clean_id_2017)
