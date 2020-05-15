# Read-in MALDI-TOF spectra in Brukerflex format 
# and write files to easily accessible text files.

# C. Weis Feb 2020

library("MALDIquant")
library("MALDIquantForeign")
library(stringr)

options(warn=0)

#########################################
# define paths
#########################################

SINK_FILE = paste('./log/DRIAMS-F_2019_preprocessed_', Sys.Date(), '.log', sep='')

sink(SINK_FILE, append=FALSE, split=FALSE)

FID_DIR = '/links/groups/borgwardt/Data/maldi_repetitions/sa_ec/ksbl_ec_sa_spectra/'
OUT_DIR = '/links/groups/borgwardt/Data/DRIAMS/DRIAMS-F/preprocessed/2019/'


list_files = list.files(path=FID_DIR, pattern = "fid$", recursive = TRUE)
num_files = length(list_files)

num_processed = 0
num_noid = 0

print(num_files)


#########################################
# go through files and process
#########################################

for (j in 1:length(list_files)){

    filename=paste(FID_DIR,list_files[j], sep="")
    cat(c("\n", as.character(j), filename), sep="\n")

    # get fileid
    spl = unlist(strsplit(list_files[j], "[/]"))
    idx = grepl('^[0-9]_[A-Z]([0-9]|[0-9][0-9])$',spl)
    fileid = spl[which(idx==TRUE)-1]
    print(fileid)

    if (nchar(fileid)!=36){
        print('Length filid != 36')
        print(fileid)
        num_noid = num_noid+1
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

    # Trim to desired mz range
    spectra = trim(spectra[[1]], range=c(2000,20000))
    spectraMatrix = data.frame(mass(spectra),intensity(spectra))

    # create output filename
    out_filename = paste(OUT_DIR, fileid, '.txt', sep="")
    print(out_filename)

    # write file
    file_con = file(out_filename, open="wt")
    writeLines(paste("# ", filename), file_con)
    writeLines(paste("# ", fileid), file_con)
    write.table(spectraMatrix, file_con, sep=" ", row.names=FALSE)
    num_processed = num_processed+1
    flush(file_con)
    close(file_con)
}


print("number of files processed:")
print(num_processed)
print("number of no ID:")
print(num_noid)
print("Program finished!")

sink()
