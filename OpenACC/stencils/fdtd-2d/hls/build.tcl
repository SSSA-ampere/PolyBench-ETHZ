
# HW-task
set num_slots 1
set name "hls_design"
set device {xc7z020clg400-1}
set period 10
set sim 1 

open_project -reset $name

# HW-task files
add_files srcs/hw_task.cpp
add_files srcs/hw_mod.cpp
add_file -tb srcs/test.cpp

set i 0
while {$i < $num_slots} {
  set_top hw_task_$i
  open_solution -reset "solution_$i"
  set_part ${device} -tool vivado
  create_clock -period ${period} -name default

  if {$sim != 0} {
    csim_design
  }
  csynth_design
  if {$sim != 0} {
    cosim_design
  }
  export_design -format ip_catalog -description "${name}_${i}" -display_name "${name}_${i}"
  incr i
}

close_project
exit

