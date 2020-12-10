# Aline Cuénod
# This scripts reads out the acquisition date of each maldi tof mass spectra and creates a csv file with the columns 'brukercode' and 'acquisition_date'. It requires the folowing two arguments:
# (i) input dir to were the spectra are located
# (ii) output dir to were the csv file should be written to

#load packages
library('MALDIquant')
library('MALDIquantForeign')

args = commandArgs(trailingOnly=TRUE)

#import spectra per year
spectra <-importBrukerFlex(args[1])

#Loop through every spectrum and read acquisition date from metadata
brukercode<-list()
measuring_date<-list()
for (i in 1:length(spectra)){
  print(paste(i,spectra[[i]]@metaData$sampleName))
  brukercode<-append(brukercode, spectra[[i]]@metaData$sampleName)
  measuring_date<-append(measuring_date, spectra[[i]]@metaData$acquisitionDate)
}

#reformat data
measuring_date_date<-gsub('(\\d{4}\\-\\d{2}\\-\\d{2})(\\T)(\\d{2}\\:\\d{2}\\:\\d{2})(.*)', '\\1', measuring_date)
measuring_date_time<-gsub('(\\d{4}\\-\\d{2}\\-\\d{2})(\\T)(\\d{2}\\:\\d{2}\\:\\d{2})(.*)', '\\3', measuring_date)

# Build datafram
date_df<-data.frame("brukercode" = unlist(brukercode), "acquisition_date" = measuring_date_date, "acquisition_time" = measuring_date_time)

#Write output
write.csv(date_df, args[2], row.names = FALSE)
