package com.database.mico_database.config;

import java.io.File;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Makes the default datasource port point at the MySQL instance on the group server.
 * The database is intentionally not exposed directly on the network: SSH forwards
 * localhost:13306 to 10.31.2.52:3306 before Spring creates the datasource.
 */
public final class RemoteDatabaseTunnel {

    private static final String DEFAULT_HOST = "10.31.2.52";
    private static final String DEFAULT_USER = "ljy";
    private static final int DEFAULT_LOCAL_PORT = 13306;
    private static final int REMOTE_MYSQL_PORT = 3306;

    private RemoteDatabaseTunnel() {
    }

    public static void ensureAvailable() {
        if (!isEnabled()) {
            return;
        }

        int localPort = integerEnv("MYSQL_PORT", DEFAULT_LOCAL_PORT);
        if (localPort != DEFAULT_LOCAL_PORT) {
            // A deployment can override MYSQL_PORT=3306 and connect locally on the server.
            return;
        }

        if (isPortOpen(localPort)) {
            System.out.println("Remote MySQL SSH tunnel is already available on localhost:" + localPort);
            return;
        }

        String keyFile = env("REMOTE_DB_SSH_KEY",
                System.getProperty("user.home") + File.separator + ".ssh" + File.separator + "mico_remote_ed25519");
        if (!Files.isRegularFile(Paths.get(keyFile))) {
            throw new IllegalStateException("Remote database SSH key is missing: " + keyFile);
        }

        String remoteHost = env("REMOTE_DB_SSH_HOST", DEFAULT_HOST);
        String remoteUser = env("REMOTE_DB_SSH_USER", DEFAULT_USER);
        Process process = startTunnel(keyFile, remoteUser, remoteHost, localPort);

        long deadline = System.currentTimeMillis() + 8000L;
        while (System.currentTimeMillis() < deadline) {
            if (isPortOpen(localPort)) {
                Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
                    @Override
                    public void run() {
                        process.destroy();
                    }
                }, "remote-mysql-tunnel-shutdown"));
                System.out.println("Remote MySQL SSH tunnel established on localhost:" + localPort);
                return;
            }
            if (!process.isAlive()) {
                break;
            }
            try {
                Thread.sleep(200L);
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                process.destroy();
                throw new IllegalStateException("Interrupted while starting remote database tunnel", ex);
            }
        }

        process.destroy();
        throw new IllegalStateException("Unable to establish remote MySQL SSH tunnel on localhost:" + localPort);
    }

    private static Process startTunnel(String keyFile, String remoteUser, String remoteHost, int localPort) {
        List<String> command = new ArrayList<String>();
        command.add(isWindows() ? "ssh.exe" : "ssh");
        command.add("-N");
        command.add("-i");
        command.add(keyFile);
        command.add("-o");
        command.add("BatchMode=yes");
        command.add("-o");
        command.add("ExitOnForwardFailure=yes");
        command.add("-o");
        command.add("ServerAliveInterval=30");
        command.add("-o");
        command.add("ServerAliveCountMax=3");
        command.add("-o");
        command.add("StrictHostKeyChecking=accept-new");
        command.add("-L");
        command.add("127.0.0.1:" + localPort + ":127.0.0.1:" + REMOTE_MYSQL_PORT);
        command.add(remoteUser + "@" + remoteHost);

        try {
            return new ProcessBuilder(command)
                    .redirectErrorStream(true)
                    .redirectOutput(ProcessBuilder.Redirect.INHERIT)
                    .start();
        } catch (IOException ex) {
            throw new IllegalStateException("Unable to start SSH for the remote MySQL tunnel", ex);
        }
    }

    private static boolean isEnabled() {
        return Boolean.parseBoolean(env("REMOTE_DB_TUNNEL_ENABLED", "true"));
    }

    private static boolean isPortOpen(int port) {
        try {
            Socket socket = new Socket();
            try {
                socket.connect(new InetSocketAddress("127.0.0.1", port), 300);
                return true;
            } finally {
                socket.close();
            }
        } catch (IOException ex) {
            return false;
        }
    }

    private static int integerEnv(String name, int defaultValue) {
        try {
            return Integer.parseInt(env(name, String.valueOf(defaultValue)));
        } catch (NumberFormatException ex) {
            throw new IllegalStateException("Invalid " + name + " value", ex);
        }
    }

    private static String env(String name, String defaultValue) {
        String value = System.getenv(name);
        return value == null || value.trim().isEmpty() ? defaultValue : value.trim();
    }

    private static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase().contains("win");
    }
}
