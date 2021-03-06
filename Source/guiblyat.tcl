#############################################################################
# Generated by PAGE version 5.1
#  in conjunction with Tcl version 8.6
#  May 05, 2020 01:08:37 AM IST  platform: Windows NT
set vTcl(timestamp) ""


if {!$vTcl(borrow) && !$vTcl(template)} {

set vTcl(actual_gui_bg) #d9d9d9
set vTcl(actual_gui_fg) #000000
set vTcl(actual_gui_analog) #ececec
set vTcl(actual_gui_menu_analog) #ececec
set vTcl(actual_gui_menu_bg) #d9d9d9
set vTcl(actual_gui_menu_fg) #000000
set vTcl(complement_color) #d9d9d9
set vTcl(analog_color_p) #d9d9d9
set vTcl(analog_color_m) #ececec
set vTcl(active_fg) #000000
set vTcl(actual_gui_menu_active_bg)  #ececec
set vTcl(pr,menufgcolor) #000000
set vTcl(pr,menubgcolor) #d9d9d9
set vTcl(pr,menuanalogcolor) #ececec
set vTcl(pr,treehighlight) firebrick
set vTcl(pr,autoalias) 1
set vTcl(pr,relative_placement) 1
set vTcl(mode) Relative
}




proc vTclWindow.top43 {base} {
    global vTcl
    if {$base == ""} {
        set base .top43
    }
    if {[winfo exists $base]} {
        wm deiconify $base; return
    }
    set top $base
    ###################
    # CREATING WIDGETS
    ###################
    vTcl::widgets::core::toplevel::createCmd $top -class Toplevel \
        -menu "$top.m44" -background #fbb7d7 \
        -highlightbackground $vTcl(actual_gui_bg) -highlightcolor black 
    wm focusmodel $top passive
    wm geometry $top 583x375+650+150
    update
    # set in toplevel.wgt.
    global vTcl
    global img_list
    set vTcl(save,dflt,origin) 0
    wm maxsize $top 1924 1030
    wm minsize $top 148 1
    wm overrideredirect $top 0
    wm resizable $top 1 1
    wm deiconify $top
    wm title $top "OCR"
    vTcl:DefineAlias "$top" "ocr" vTcl:Toplevel:WidgetProc "" 1
    set vTcl(real_top) {}
    vTcl:withBusyCursor {
    set site_3_0 $top.m44
    menu $site_3_0 \
        -activebackground SystemHighlight \
        -activeforeground SystemHighlightText \
        -background $vTcl(pr,menubgcolor) -font TkDefaultFont \
        -foreground $vTcl(pr,menufgcolor) -tearoff 0 
    $site_3_0 add cascade \
        -menu "$site_3_0.men53" -activebackground $vTcl(analog_color_m) \
        -activeforeground #000000 -background $vTcl(pr,menubgcolor) \
        -command {{}} -font TkMenuFont -foreground $vTcl(pr,menufgcolor) \
        -label File 
    menu $site_3_0.men53 \
        -activebackground #f9f9f9 -activeforeground black \
        -background $vTcl(pr,menubgcolor) -font {-family {Segoe UI} -size 9} \
        -foreground black -tearoff 0 
    $site_3_0.men53 add command \
        -activebackground $vTcl(analog_color_m) -activeforeground #000000 \
        -background $vTcl(pr,menubgcolor) -command {{}} -font TkMenuFont \
        -foreground $vTcl(pr,menufgcolor) -label {Save Result Text} 
    $site_3_0.men53 add command \
        -activebackground $vTcl(analog_color_m) -activeforeground #000000 \
        -background $vTcl(pr,menubgcolor) -command {{}} -font TkMenuFont \
        -foreground $vTcl(pr,menufgcolor) -label {Save Result Image} 
    $site_3_0.men53 add command \
        -activebackground $vTcl(analog_color_m) -activeforeground #000000 \
        -background $vTcl(pr,menubgcolor) -command {{}} -font TkMenuFont \
        -foreground $vTcl(pr,menufgcolor) -label {Save Result} 
    $site_3_0 add cascade \
        -menu "$site_3_0.men53" -activebackground $vTcl(analog_color_m) \
        -activeforeground #000000 -background $vTcl(pr,menubgcolor) \
        -command {{}} -font TkMenuFont -foreground $vTcl(pr,menufgcolor) \
        -label File -menu "$site_3_0.men54" \
        -activebackground $vTcl(analog_color_m) -activeforeground #000000 \
        -background $vTcl(pr,menubgcolor) -command {{}} -font TkMenuFont \
        -foreground $vTcl(pr,menufgcolor) -label Help 
    menu $site_3_0.men54 \
        -activebackground #f9f9f9 -activeforeground black \
        -background $vTcl(pr,menubgcolor) -font {-family {Segoe UI} -size 9} \
        -foreground black -tearoff 0 
    $site_3_0 add command \
        -activebackground $vTcl(analog_color_m) -activeforeground #000000 \
        -background $vTcl(pr,menubgcolor) \
        -command {#lambda : destroy_window()} -font TkMenuFont \
        -foreground $vTcl(pr,menufgcolor) -label Exit 
    button $top.but49 \
        -activebackground $vTcl(analog_color_m) -activeforeground #000000 \
        -background $vTcl(actual_gui_bg) -disabledforeground #a3a3a3 \
        -font TkDefaultFont -foreground $vTcl(actual_gui_fg) \
        -highlightbackground $vTcl(actual_gui_bg) -highlightcolor black \
        -pady 0 -text Back 
    vTcl:DefineAlias "$top.but49" "Button2" vTcl:WidgetProc "ocr" 1
    button $top.but51 \
        -activebackground $vTcl(analog_color_m) -activeforeground #000000 \
        -background $vTcl(actual_gui_bg) -disabledforeground #a3a3a3 \
        -font TkDefaultFont -foreground $vTcl(actual_gui_fg) \
        -highlightbackground $vTcl(actual_gui_bg) -highlightcolor black \
        -pady 0 -text Close 
    vTcl:DefineAlias "$top.but51" "Button3" vTcl:WidgetProc "ocr" 1
    message $top.mes52 \
        -background #ffffff -font TkDefaultFont \
        -foreground $vTcl(actual_gui_fg) \
        -highlightbackground $vTcl(actual_gui_bg) -highlightcolor black \
        -width 576 
    vTcl:DefineAlias "$top.mes52" "Message1" vTcl:WidgetProc "ocr" 1
    canvas $top.can44 \
        -background #fbb7d7 -borderwidth 2 -closeenough 1.0 -height 183 \
        -insertbackground black -relief ridge -selectbackground #c4c4c4 \
        -selectforeground black -width 198 
    vTcl:DefineAlias "$top.can44" "Canvas1" vTcl:WidgetProc "ocr" 1
    canvas $top.can45 \
        -background #fbb7d7 -borderwidth 2 -closeenough 1.0 -height 183 \
        -insertbackground black -relief ridge -selectbackground #c4c4c4 \
        -selectforeground black -width 188 
    vTcl:DefineAlias "$top.can45" "Canvas2" vTcl:WidgetProc "ocr" 1
    ###################
    # SETTING GEOMETRY
    ###################
    place $top.but49 \
        -in $top -x 0 -relx 0.167 -y 0 -rely 0.85 -width 126 -relwidth 0 \
        -height 33 -relheight 0 -anchor nw -bordermode ignore 
    place $top.but51 \
        -in $top -x 0 -relx 0.6 -y 0 -rely 0.853 -width 126 -relwidth 0 \
        -height 33 -relheight 0 -anchor nw -bordermode ignore 
    place $top.mes52 \
        -in $top -x 0 -relx 0.017 -y 0 -rely 0.613 -width 0 -relwidth 0.961 \
        -height 0 -relheight 0.179 -anchor nw -bordermode ignore 
    place $top.can44 \
        -in $top -x 0 -relx 0.103 -y 0 -rely 0.08 -width 0 -relwidth 0.34 \
        -height 0 -relheight 0.488 -anchor nw -bordermode ignore 
    place $top.can45 \
        -in $top -x 0 -relx 0.566 -y 0 -rely 0.08 -width 0 -relwidth 0.322 \
        -height 0 -relheight 0.488 -anchor nw -bordermode ignore 
    } ;# end vTcl:withBusyCursor 

    vTcl:FireEvent $base <<Ready>>
}

set btop ""
if {$vTcl(borrow)} {
    set btop .bor[expr int([expr rand() * 100])]
    while {[lsearch $btop $vTcl(tops)] != -1} {
        set btop .bor[expr int([expr rand() * 100])]
    }
}
set vTcl(btop) $btop
Window show .
Window show .top43 $btop
if {$vTcl(borrow)} {
    $btop configure -background plum
}

