package com.database.mico_database;

import com.database.mico_database.config.RemoteDatabaseTunnel;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class MicoDatabaseApplication {

    public static void main(String[] args) {
        RemoteDatabaseTunnel.ensureAvailable();
        SpringApplication.run(MicoDatabaseApplication.class, args);
    }

}
