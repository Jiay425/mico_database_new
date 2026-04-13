package com.database.mico_database.model;

import lombok.Data;

@Data
public class SysUser {
    private Long id;
    private String username;
    private String password;
    private String role;
    private boolean enabled;
}