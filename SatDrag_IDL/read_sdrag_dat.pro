;+
; :Description:
;    Read in satellite drag data set.
;
;
;
; :Keywords:
;    date_min
;    date_max
;    sd_fn
;    slist
;
; :Author: krmurph1
;-
function read_sdrag_dat, $
  date_min=date_min, $
  date_max=date_max, $
  sd_fn = sd_fn, $
  slist = slist
  
  
  if keyword_set(sd_fn) then sd_fn=sd_fn else sd_fn='D:\data\VL_sdrag\combined_data_all_reduced_omni.csv'
  
  fs = file_search(sd_fn, count=fc)
  
  if fc eq 0 then return, 0
  
  fl = file_lines(fs[0])
  
  openr, inlun, sd_fn, /get_lun
  line=' ' 
  
  readf,inlun,line
  
  t_th = dblarr(fl)
  d400 = fltarr(fl)
  f107 = d400
  s_la = d400
  s_mg = d400
  s_ir = d400
  
  i=0L
  while not eof(inlun) do begin
    readf,inlun,line
    dat = strsplit(line,',',/extract)
    
    tt = time_double(dat[0],tformat='YYYY-MM-DD hh:mm:ss')
    
    t_th[i] = tt
    d400[i] = float(dat[12])
    f107[i] = float(dat[19])
    s_la[i] = float(dat[22])
    s_mg[i] = float(dat[23])
    s_ir[i] = float(dat[26])
    
    if i mod 100000 eq 0 then print, i     
    i++ 
  endwhile
  
  t_i = sort(t_th[0:i-1], /L64)
  t_th = t_th[t_i]
  d400 = d400[t_i]
  f107 = f107[t_i]
  s_la = s_la[t_i]
  s_mg = s_mg[t_i]
  s_ir = s_ir[t_i]
  
  ; if you have a storm list
  ; check if the date is a storm time
  ; and only return storm time data
  if keyword_set(slist) then begin
    st = intarr(i)
    st_i = [ ] 
    st_st = time_double(slist.t1)
    st_en = time_double(slist.t3)
    for j=0L, st_st.length-1 do begin
      gd = where(t_th ge st_st[j] and t_th le st_en[j], c)
      
      if c gt 0 then begin
        st[gd] = 1
        st_i = [st_i,gd]
      endif
    endfor
    
    t_th = t_th[st_i]
    d400 = d400[st_i]
    f107 = f107[st_i]
    s_la = s_la[st_i]
    s_mg = s_mg[st_i]
    s_ir = s_ir[st_i]  
  endif
  
  r_dat = {t_th:t_th, d400:d400, f107:f107, s_la:s_la, s_mg:s_mg, s_ir:s_ir}
  
  return, r_dat

end


; main
ifile = 'D:\data\Storms\sdrag\storms_drag_epochs.txt'
;read storm list
slist = read_storm_list(ifile)
;read sdrag data
a = read_sdrag_dat(slist=slist)


end
  
  