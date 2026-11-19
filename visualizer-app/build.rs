fn main() {
    if cfg!(target_os = "windows") {
        let mut res = winres::WindowsResource::new();
        // Make sure the path is correct relative to this build.rs file
        res.set_icon("../assets/icon.ico"); 
        res.set("ProductName", "SABI NEIRO");
        res.set("FileDescription", "SABI NEIRO");
        res.compile().unwrap();
    }
}
