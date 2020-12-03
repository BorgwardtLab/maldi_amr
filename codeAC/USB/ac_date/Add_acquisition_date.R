# Aline Cuénod
# 2020
# This script adds to every IDRES File of the DRAIMS-A dataset two columns: 'acquisition_date' and 'acquisition_time', indicating when the mass spectra was acquired
# Acquisition date and time were extracted from fid files using the script "read_acquisition_time_from_fid.R"

# 2015
ID_RES_2015<-read.csv('./Stratification/2015-01-12_IDRES_AB_not_summarised.csv')
ac_date2015<-read.csv('./ac_date/acquisition_dates_2015.csv')
ac_date2015$brukercode<-gsub("\\_","-", ac_date2015$brukercode)

# remove entries with duplicate brukercode, these are repeated measurements of the same spot
# Choose the later acquisition time, as this is most likely the better / non-empty spectrum
ac_date2015<-ac_date2015[!duplicated(ac_date2015),]
ac_date2015<-ac_date2015 %>% 
  group_by(brukercode) %>%
  filter(acquisition_date == max(acquisition_date)) %>%
  filter(acquisition_time == max(acquisition_time))


ID_RES_2015<-merge(ID_RES_2015, ac_date2015, by.x = 'code', by.y = 'brukercode', all.x = T)

write.csv(ID_RES_2015, './ac_date/2015-01-12_IDRES_AB_not_summarised.csv', row.names = F, quote = F)

# 2016
ID_RES_2016<-read.csv('./Stratification/2016-01-12_IDRES_AB_not_summarised.csv')
ac_date2016<-read.csv('./ac_date/acquisition_dates_2016.csv')
ac_date2016$brukercode<-gsub("\\_","-", ac_date2016$brukercode)

# remove entries with duplicate brukercode, these are repeated measurements of the same spot
# Choose the later acquisition time, as this is most likely the better / non-empty spectrum
ac_date2016<-ac_date2016[!duplicated(ac_date2016),]
ac_date2016<-ac_date2016 %>% 
  group_by(brukercode) %>%
  filter(acquisition_date == max(acquisition_date)) %>%
  filter(acquisition_time == max(acquisition_time))

ID_RES_2016<-merge(ID_RES_2016, ac_date2016, by.x = 'code', by.y = 'brukercode', all.x = T)
write.csv(ID_RES_2016, './ac_date/2016-01-12_IDRES_AB_not_summarised.csv', row.names = F, quote = F)

# 2017
ID_RES_2017<-read.csv('./Stratification/2017-01-12_IDRES_AB_not_summarised.csv')

# m1
ac_date2017_m1<-read.csv('./ac_date/acquisition_dates_2017_m1.csv')
ac_date2017_m1$brukercode<-gsub("\\_","-", ac_date2017_m1$brukercode)
ac_date2017_m1$brukercode<-paste0(ac_date2017_m1$brukercode, '_MALDI1')

ac_date2017_m2<-read.csv('./ac_date/acquisition_dates_2017_m2.csv')
ac_date2017_m2$brukercode<-gsub("\\_","-", ac_date2017_m2$brukercode)
ac_date2017_m2$brukercode<-paste0(ac_date2017_m2$brukercode, '_MALDI2')

#combine m1 and m2
ac_date2017<-merge(ac_date2017_m1, ac_date2017_m2, by = intersect(colnames(ac_date2017_m1), colnames(ac_date2017_m2)), all = T)

# all which are missing are either empty ('no peaks found') or do not have a brukercode
# check<-ID_RES_2017[ID_RES_2017$code %in% setdiff(ID_RES_2017$code, ac_date2017$brukercode), ]


# remove entries with duplicate brukercode, these are repeated measurements of the same spot
# Choose the later acquisition time, as this is most likely the better / non-empty spectrum
ac_date2017<-ac_date2017[!duplicated(ac_date2017),]
ac_date2017<-ac_date2017 %>% 
  group_by(brukercode) %>%
  filter(acquisition_date == max(acquisition_date)) %>%
  filter(acquisition_time == max(acquisition_time))


ID_RES_2017<-merge(ID_RES_2017, ac_date2017, by.x = 'code', by.y = 'brukercode', all.x = T)
write.csv(ID_RES_2017, './ac_date/2017-01-12_IDRES_AB_not_summarised.csv')

# 2018
ID_RES_2018<-read.csv('./Stratification/2018_01-08_IDRES_AB_not_summarised.csv', row.names = F, quote = F)

# m1
ac_date2018_m1<-read.csv('./ac_date/acquisition_dates_2018_m1.csv')
ac_date2018_m1$brukercode<-gsub("\\_","-", ac_date2018_m1$brukercode)
ac_date2018_m1$brukercode<-paste0(ac_date2018_m1$brukercode, '_MALDI1')

ac_date2018_m2<-read.csv('./ac_date/acquisition_dates_2018_m2.csv')
ac_date2018_m2$brukercode<-gsub("\\_","-", ac_date2018_m2$brukercode)
ac_date2018_m2$brukercode<-paste0(ac_date2018_m2$brukercode, '_MALDI2')

#combine m1 and m2
ac_date2018<-merge(ac_date2018_m1, ac_date2018_m2, by = intersect(colnames(ac_date2018_m1), colnames(ac_date2018_m2)), all = T)

# all which are missing are either empty ('no peaks found') or do not have a brukercode
# check<-ID_RES_2018[ID_RES_2018$code %in% setdiff(ID_RES_2018$code, ac_date2018$brukercode), ]


# remove entries with duplicate brukercode, these are repeated measurements of the same spot
# Choose the later acquisition time, as this is most likely the better / non-empty spectrum
ac_date2018<-ac_date2018[!duplicated(ac_date2018),]
ac_date2018<-ac_date2018 %>% 
  group_by(brukercode) %>%
  filter(acquisition_date == max(acquisition_date)) %>%
  filter(acquisition_time == max(acquisition_time))


ID_RES_2018<-merge(ID_RES_2018, ac_date2018, by.x = 'code', by.y = 'brukercode', all.x = T, row.names = F, quote = F)
write.csv(ID_RES_2018, './ac_date/2018_01-08_IDRES_AB_not_summarised.csv')


