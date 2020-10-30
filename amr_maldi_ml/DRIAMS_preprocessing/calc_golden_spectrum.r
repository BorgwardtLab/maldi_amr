library(MALDIquant)
library(MALDIquantForeign)
library(methods)
library(stringr)

species = 'Escherichia_coli'
#species = 'Klebsiella_pneumoniae'
#species = 'Staphylococcus_aureus'

RAW_DIR = '/links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/raw/'
OUT_DIR = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/amr_maldi_ml/DRIAMS_preprocessing/reference_peaks/'

# extract list of codes matching to the given species
species_file = paste(OUT_DIR, 'codes_per_species_', species, '.txt', sep='')
species_codes = scan(species_file, what="", skip=1, sep="\n")

# read in all files in 'raw' spectrum folder
files = list.files(path=RAW_DIR, pattern = "txt$", recursive=TRUE)

# create empty lists to fill up with spectra
spectra = vector('list', length(files))

# ------------------------------------
# (1)   Create list with all raw spectra of species 
# ------------------------------------
for (i in 1:length(files)){
#for (i in 1:300){
    file = files[i]
    code = rev(unlist(strsplit(file, "[/]")))[1]
    code = unlist(strsplit(code, "[.]"))[1]

    filename = paste(RAW_DIR,file,sep="")
    
    # skip if spectrum is in codes_per_species file
    if (!(code %in% species_codes)){
        next
    }
    
    # read in file as MassSpec class
    tryCatch(
        expr = {
            spectrum = importTxt(filename, 
                                 header = TRUE, 
                                 sep = " ", 
                                 comment.char = "#")

            # concatenate all spectra to calculate reference spectrum
            spectra[[i]] = spectrum[[1]]
            names(spectra)[i] = toString(code) 
        },
        error = function(e){
            print(filename)
            message('Caught an error!')
            print(e)
        })
}


# ------------------------------------
# (2)   Remove missing spectra 
# ------------------------------------

# many spectra are NULL, as all non-target-species positions are NULL
table(sapply(spectra, length))
spectra = spectra[lapply(spectra, length)!=0]
print(any(sapply(spectra, isEmpty)))
table(sapply(spectra, length))

# ------------------------------------
# (3)   Preprocess spectra and calculate golden peaks 
# ------------------------------------

# preprocessing steps / should be the same as used later for preprocessing
spectra = transformIntensity(spectra, method="sqrt")
spectra = smoothIntensity(spectra, method="SavitzkyGolay", halfWindowSize=10)
spectra = removeBaseline(removeBaseline(spectra, method="SNIP", iterations=20))
spectra = calibrateIntensity(spectra, method="TIC")
table(sapply(spectra, length))

# peak calling / specific to this script
min_freq_peaks = 0.90
tolerance = 0.004
peaks = detectPeaks(spectra, method="MAD", halfWindowSize=20, SNR=3)
print(length(peaks))
reference_peaks = referencePeaks(peaks, 
                                 method='strict', 
                                 minFrequency=min_freq_peaks, 
                                 tolerance=tolerance)

print(reference_peaks)

# write to output
data_matrix = data.frame(mass(reference_peaks), intensity(reference_peaks))

filename = paste(species,min_freq_peaks,tolerance,'ref_peaks.txt',sep='_')
filename = paste(OUT_DIR, filename, sep='') 
f = file(filename, open="wt")
writeLines(paste("# ", toString(species)), f)
write.table(data_matrix, f, sep=" ", row.names=FALSE)
flush(f)
close(f)
