;+
; :Description:
;    Run normalized superposed epoch analysis 
;    for storms on geomagnetic, solar, and 
;    satellite drag data.
;
;
;
;
;
; :Author: krmurph1
;-
pro sdrag_epoch

  fixplot

  ifile = 'D:\data\Storms\sdrag\storms_drag_epochs.txt'


  ;read storm list
  slist = read_storm_list(ifile)
  date_min = time_string(min(time_double(slist.t1)),tformat='YYYY-01-01')
  date_max = time_string(max(time_double(slist.t3)),tformat='YYYY-12-31')

  ;read in omni data over storm periods
  om_dat = read_omni_1hr(date_min=date_min,date_max=date_max)

  ;read in the satellite drag dataset
  sd_dat = read_sdrag_dat(slist=slist)

  ;sort omni data into superposed epoch analysis
  ; first phase is 30 hours long
  ; second is 120 hours long
  ; bins size is 1 hour
  ;
  ; the structure tags from omni that we are sorting are
  ;['V','FP','BZ_GSM','DST','AE']
  ; and the structure tag that holds the time for variables in themis format is
  ;date_themis
  epoch_o1hr = get_epoch_data(om_dat,time_double(slist.t1),time_double(slist.t3),time_double(slist.t2), 30,120,1,['V','FP','BZ_GSM','DST','AE','KP'],t_tags='date_themis')
  epoch_sd = get_epoch_data(sd_dat,time_double(slist.t1),time_double(slist.t3),time_double(slist.t2), 30,120,1,['D400','F107','S_LA','S_MG','S_IR'],t_tags='t_th')
  

  window, 9, xsize = 550, ysize = 900
  !p.charsize = 1.75
  !x.style = 1
  !y.style = 1
  !p.thick = 1.
  !y.margin = [0.,0.5]
  !y.omargin = [5,2]

  !p.multi = [0,1,6]
  !x.omargin = [0,0]
  !x.margin = [15,10]

  vplot2, epoch_o1hr.v.ep1, epoch_o1hr.v.ep2, xtickf='no_ticks', ytitle = 'V!DSW!N!C!Ckm/s', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange, title = ifile
  oplot, [0,0],!y.crange, linestyle = 2
  vplot2, epoch_o1hr.fp.ep1, epoch_o1hr.fp.ep2, xtickf='no_ticks', ytitle = 'Pressure!CnPa', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange
  oplot, [0,0],!y.crange, linestyle = 2
  vplot2, epoch_o1hr.bz_gsm.ep1, epoch_o1hr.bz_gsm.ep2, xtickf='no_ticks', ytitle = 'B!DZ!N!C!CnT', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange
  oplot, [0,0],!y.crange, linestyle = 2
  vplot2, epoch_o1hr.dst.ep1, epoch_o1hr.dst.ep2, xtickf='no_ticks', ytitle = 'Sym H!CnT', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange
  oplot, [0,0],!y.crange, linestyle = 2
  vplot2, epoch_o1hr.kp.ep1, epoch_o1hr.kp.ep2, xtickf='no_ticks', ytitle = 'Kp!CnT', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange
  oplot, [0,0],!y.crange, linestyle = 2
  vplot2, epoch_o1hr.ae.ep1, epoch_o1hr.ae.ep2, ytitle = 'AE!CnT', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange
  oplot, [0,0],!y.crange, linestyle = 2
  
  
  window, 10, xsize = 550, ysize = 700
  !p.multi = [0,1,5]

  vplot2, epoch_sd.d400.ep1, epoch_sd.d400.ep2, xtickf='no_ticks', ytitle = 'Density 400km', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange, title = ifile
  oplot, [0,0],!y.crange, linestyle = 2
  vplot2, epoch_sd.f107.ep1, epoch_sd.f107.ep2, xtickf='no_ticks', ytitle = 'F 10.7', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange
  oplot, [0,0],!y.crange, linestyle = 2
  vplot2, epoch_sd.s_la.ep1, epoch_sd.s_la.ep2, xtickf='no_ticks', ytitle = 'Solar Lyman-Alpha', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange
  oplot, [0,0],!y.crange, linestyle = 2
  vplot2, epoch_sd.s_mg.ep1, epoch_sd.s_mg.ep2, xtickf='no_ticks', ytitle = 'Mg Index', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange
  oplot, [0,0],!y.crange, linestyle = 2
  vplot2, epoch_sd.s_ir.ep1, epoch_sd.s_ir.ep2, ytitle = 'Irridiance', xcharsize = xc, ycharsize=yc,ylog=0, xrange=xrange
  oplot, [0,0],!y.crange, linestyle = 2

  stop
end