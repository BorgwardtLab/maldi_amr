# Aline Cuénod
# 2021
# This script adds to every IDRES File of the DRAIMS-A dataset the columns: 'workstation', indicating on which workstation of the routine diagnostics laboratory the samples were processed

# 2015
ID_RES_2015<-read.csv('./ac_date/2015-01-12_IDRES_AB_not_summarised.csv', sep = ';')
ID_RES_2015['station_temp']<-gsub('(2015)(\\d{1})(\\d{5})', '\\2', ID_RES_2015$TAGESNUMMER)
table(ID_RES_2015$station_temp)
ID_RES_2015['workstation']<-ifelse(ID_RES_2015$station_temp == '1', 'Blood', 
                                     ifelse(ID_RES_2015$station_temp == '2', 'Genital', 
                                            ifelse(ID_RES_2015$station_temp == '3', 'Stool',
                                                   ifelse(ID_RES_2015$station_temp == '4', 'Varia',
                                                          ifelse(ID_RES_2015$station_temp == '5', 'Respiratory',
                                                                 ifelse(ID_RES_2015$station_temp == '6', 'DeepTissue',
                                                                        ifelse(ID_RES_2015$station_temp == '7', 'Urine',
                                                                               ifelse(ID_RES_2015$station_temp == '8', 'HospitalHygiene',
                                                                                      ifelse(ID_RES_2015$station_temp == '9', 'PCR',NA)))))))))


ID_RES_2015$station_temp<-NULL
table(ID_RES_2015$workstation)
write.table(ID_RES_2015, './ac_add_workstation/2015-01-12_IDRES_AB_not_summarised.csv', row.names = F, quote = F, sep=';')

# 2016
ID_RES_2016<-read.csv('./ac_date/2016-01-12_IDRES_AB_not_summarised.csv', sep = ';')
ID_RES_2016['station_temp']<-gsub('(2016)(\\d{1})(\\d{5})', '\\2', ID_RES_2016$TAGESNUMMER)
table(ID_RES_2016$station_temp)
ID_RES_2016['workstation']<-ifelse(ID_RES_2016$station_temp == '1', 'Blood', 
                                   ifelse(ID_RES_2016$station_temp == '2', 'Genital', 
                                          ifelse(ID_RES_2016$station_temp == '3', 'Stool',
                                                 ifelse(ID_RES_2016$station_temp == '4', 'Varia',
                                                        ifelse(ID_RES_2016$station_temp == '5', 'Respiratory',
                                                               ifelse(ID_RES_2016$station_temp == '6', 'DeepTissue',
                                                                      ifelse(ID_RES_2016$station_temp == '7', 'Urine',
                                                                             ifelse(ID_RES_2016$station_temp == '8', 'HospitalHygiene',
                                                                                    ifelse(ID_RES_2016$station_temp == '9', 'PCR',NA)))))))))


ID_RES_2016$station_temp<-NULL
table(ID_RES_2016$workstation)

write.table(ID_RES_2016, './ac_add_workstation/2016-01-12_IDRES_AB_not_summarised.csv', sep = ";", row.names = F, quote = F)

# 2017
ID_RES_2017<-read.csv('./ac_date/2017-01-12_IDRES_AB_not_summarised.csv', sep = ';')
ID_RES_2017['station_temp']<-gsub('(2017)(\\d{1})(\\d{5})', '\\2', ID_RES_2017$TAGESNUMMER)
table(ID_RES_2017$station_temp)
ID_RES_2017['workstation']<-ifelse(ID_RES_2017$station_temp == '1', 'Blood', 
                                   ifelse(ID_RES_2017$station_temp == '2', 'Genital', 
                                          ifelse(ID_RES_2017$station_temp == '3', 'Stool',
                                                 ifelse(ID_RES_2017$station_temp == '4', 'Varia',
                                                        ifelse(ID_RES_2017$station_temp == '5', 'Respiratory',
                                                               ifelse(ID_RES_2017$station_temp == '6', 'DeepTissue',
                                                                      ifelse(ID_RES_2017$station_temp == '7', 'Urine',
                                                                             ifelse(ID_RES_2017$station_temp == '8', 'HospitalHygiene',
                                                                                    ifelse(ID_RES_2017$station_temp == '9', 'PCR',NA)))))))))


ID_RES_2017$station_temp<-NULL
table(ID_RES_2017$workstation)

write.table(ID_RES_2017, './ac_add_workstation/2017-01-12_IDRES_AB_not_summarised.csv', row.names = F, quote = F, sep=';')

# 2018
ID_RES_2018<-read.csv('./ac_date/2018_01-08_IDRES_AB_not_summarised.csv', sep=';')
ID_RES_2018['station_temp']<-gsub('(2018)(\\d{1})(\\d{5})', '\\2', ID_RES_2018$TAGESNUMMER)
table(ID_RES_2018$station_temp)
ID_RES_2018['workstation']<-ifelse(ID_RES_2018$station_temp == '1', 'Blood', 
                                   ifelse(ID_RES_2018$station_temp == '2', 'Genital', 
                                          ifelse(ID_RES_2018$station_temp == '3', 'Stool',
                                                 ifelse(ID_RES_2018$station_temp == '4', 'Varia',
                                                        ifelse(ID_RES_2018$station_temp == '5', 'Respiratory',
                                                               ifelse(ID_RES_2018$station_temp == '6', 'DeepTissue',
                                                                      ifelse(ID_RES_2018$station_temp == '7', 'Urine',
                                                                             ifelse(ID_RES_2018$station_temp == '8', 'HospitalHygiene',
                                                                                    ifelse(ID_RES_2018$station_temp == '9', 'PCR',NA)))))))))


ID_RES_2018$station_temp<-NULL
table(ID_RES_2018$workstation)
write.table(ID_RES_2018, './ac_add_workstation/2018_01-08_IDRES_AB_not_summarised.csv', row.names = F, quote = F, sep=';')


