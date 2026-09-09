//!OpenSCAD

thickness = 4;
depth = 210;
width = 280;
camera_dist = 400;
n_rows = 3;
n_cols = 4;
height = 50;

intersection() {
  union(){
    for (i = [0 : abs(1) : n_cols]) {
      x_shift = i * (width / n_cols) - width / 2;
      translate([x_shift, 0, 0]){
        rotate([0, (-atan(x_shift / camera_dist)), 0]){
          scale([1.1, 1.1, 2]){
            cube([thickness, depth, height], center=true);
          }
        }
      }
    }

    for (i = [0 : abs(1) : n_rows]) {
      y_shift = i * (depth / n_rows) - depth / 2;
      translate([0, y_shift, 0]){
        rotate([(atan(y_shift / camera_dist)), 0, 0]){
          scale([1.1, 1.1, 2]){
            cube([width, thickness, height], center=true);
          }
        }
      }
    }

  }

  scale([1.1, 1.1, 1]){
    cube([width, depth, height], center=true);
  }

}
