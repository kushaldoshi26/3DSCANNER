import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: { target: 'es2022', chunkSizeWarningLimit: 800, rollupOptions: {input: {main:'index.html',workspace:'workspace.html',projects:'projects.html',device:'device.html',about:'about.html',profile:'profile.html',login:'login.html',register:'register.html'}} },
});
