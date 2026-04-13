package com.database.mico_database.service;

import com.database.mico_database.mapper.UserMapper;
import com.database.mico_database.model.SysUser;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

@Service
public class UserDetailsServiceImpl implements UserDetailsService {

    @Autowired
    private UserMapper userMapper;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        SysUser sysUser = userMapper.findByUsername(username);
        if (sysUser == null) {
            throw new UsernameNotFoundException("用户不存在: " + username);
        }

        // 将我们的 SysUser 转换为 Spring Security 需要的 UserDetails 对象
        return User.builder()
                .username(sysUser.getUsername())
                .password(sysUser.getPassword()) // 这里的密码必须是加密后的
                .roles(sysUser.getRole().replace("ROLE_", "")) // Spring Security 自动加前缀，这里要去掉
                .build();
    }
}