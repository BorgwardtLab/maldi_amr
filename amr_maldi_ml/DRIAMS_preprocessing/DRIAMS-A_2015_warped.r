# Read-in MALDI-TOF spectra in Brukerflex format 
# and write files to easily accessible text files.

# C. Weis Feb 2020

library("MALDIquant")
library("MALDIquantForeign")
library(stringr)

options(warn=0)

tolerance = 0.002
warping_algo = 'linear'

for (species in c('Escherichia_coli', 'Staphylococcus_aureus', 'Klebsiella_pneumoniae')){
#########################################
# define paths
#########################################

SINK_FILE = paste('./log/DRIAMS-A_2015_preprocessed_warped_',species,'_',Sys.Date(),'.log', sep='')
sink(SINK_FILE, append=FALSE, split=FALSE)

FID_DIR = '/links/groups/borgwardt/Data/ms_diagnostics/USB/2015-01-12-monthly_added_not_renamed/'
OUT_DIR = '/links/groups/borgwardt/Data/DRIAMS/DRIAMS-A/preprocessed_warped/2015/'
WARP_DIR = '/links/groups/borgwardt/Projects/maldi_tof_diagnostics/amr_maldi_ml/amr_maldi_ml/DRIAMS_preprocessing/reference_peaks/'

# read in gold spectrum for the current species
filename = paste(WARP_DIR, species, '_0.9_0.002_ref_peaks.txt', sep='')
reference_peaks = importTxt(filename,
                            header = TRUE,
                            centroided = TRUE, #import peaks
                            sep = " ",
                            comment.char = "#")
# turn list into S4
reference_peaks = reference_peaks[[1]]

# extract list of codes matching to the given species
species_file = paste(WARP_DIR, 'codes_per_species_', species, '.txt', sep='')
species_codes = scan(species_file, what="", skip=1, sep="\n")

list_files = list.files(path=FID_DIR, pattern = "fid$", recursive = TRUE)
num_files = length(list_files)

num_processed = 0
num_noid = 0


#########################################
# go through files and process
#########################################

for (j in 1:length(list_files)){

    filename=paste(FID_DIR,list_files[j],sep="")
    cat(c("\n", as.character(j), filename), sep="\n")

    # get fileid
    spl = unlist(strsplit(list_files[j], "[/]"))
    idx = grepl('^[0-9]_[A-Z]([0-9]|[0-9][0-9])$',spl)
    fileid = spl[which(idx==TRUE)-1]
    #print(fileid)

    if (nchar(fileid)!=36){
        print('Length fileid != 36')
        num_noid = num_noid+1
        next
    }

    # skip if spectrum is in codes_per_species file
    if (!(fileid %in% species_codes)){
        cat(fileid, 'not found', sep=' ')
        next
    }

    # Import fid files
    myspec = importBrukerFlex(filename, removeEmptySpectra=TRUE)

    # Skip if spectra empty
    if (length(myspec) == 0){
        print('Spectra is empty')
        num_noid = num_noid+1
        next    
    }

    # Transform intensities / variance stabilizing
    spectra = transformIntensity(myspec, method="sqrt")

    # Smooth spectra
    spectra = smoothIntensity(spectra, method="SavitzkyGolay", halfWindowSize=10)

    # Remove baseline
    spectra = removeBaseline(removeBaseline(spectra, method="SNIP", iterations=20))

    # Intensity calibration (y axis normalization)
    spectra = calibrateIntensity(spectra, method="TIC")
    
    # Gold Spectrum warping (x axis normalization)
    peaks = detectPeaks(spectra, method="MAD", halfWindowSize=20, SNR=3)
    skip_to_next = TRUE
    # use tryCatch in case of too few peaks matching with the reference
    calc_warping_function = tryCatch(
        expr = {
            warping_function = determineWarpingFunctions(peaks, 
                                    reference = reference_peaks, 
                                    tolerance = tolerance, 
                                    method = warping_algo)
            print(spectra)
            spectra = warpMassSpectra(spectra, warping_function)
            print(spectra)
            skip_to_next = FALSE
        },
        error = function(e){
            print(filename)
            print('Could not warp spectrum!')
            print(e)
        })    

    if(skip_to_next) { next }  
    print('spectra warped..')
    
    # Trim to desired mz range
    spectra = trim(spectra[[1]], range=c(2000,20000))
    spectraMatrix = data.frame(mass(spectra),intensity(spectra))


    out_filename = paste(OUT_DIR,fileid,'.txt',sep="")
    print('writing output...')
    print(out_filename)

    # write file if it doesn't exist already
    if (!file.exists(out_filename)){
    file_con <- file(out_filename, open="wt")
    writeLines(paste("# ",filename), file_con)
    writeLines(paste("# ",fileid), file_con)
    write.table(spectraMatrix,file_con,sep=" ",row.names=FALSE)
    num_processed=num_processed+1
    flush(file_con)
    close(file_con)
    } else {
    cat(out_filename,'already exists!',sep=' ')
}}

print("number of files processed:")
print(num_processed)
print("number of no ID:")
print(num_noid)
print("Program finished!")

sink()
}
