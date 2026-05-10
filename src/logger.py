class Logger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    
    def warning(self, msg):
        print(f"[WARNING] {msg}")
    
    def success(self, msg):
        print(f"[SUCCESS] {msg}")
    
    def mark(self, msg):
        print(f"[=== MARK ===] {msg} [=== MARK ===]")
    
    def error(self, msg):
        print(f"[ERROR] {msg}")




log = Logger()

if __name__ == "__main__":
    log.info('info')
    log.warning('warning')
    log.success('success')
    log.mark('mark')
