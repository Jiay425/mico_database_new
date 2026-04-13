package com.database.mico_database.mapper;

import com.database.mico_database.model.SysUser;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface UserMapper {
    @Select("SELECT * FROM sys_user WHERE username = #{username}")
    SysUser findByUsername(String username);

    // 临时增加一个插入方法，帮您生成初始账号
    @org.apache.ibatis.annotations.Insert("INSERT INTO sys_user(username, password, role) " +
            "VALUES(#{username}, #{password}, #{role})")
    void insert(SysUser user);
}