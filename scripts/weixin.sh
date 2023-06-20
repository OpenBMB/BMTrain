#!/usr/bin/env bash

usage() {
    echo "Usage: bash $0 hook msg [mentioned_list]"
    echo "all use @all"
    exit 1
}

if [ $# -ne 3 ];then
    echo $#
    usage
fi


HOOK=$1
MSG=$2
MS=$3
#创建微信机器人方法： http://wiki.in.zhihu.com/pages/viewpage.action?pageId=92312849
#sh -x script/weixin.sh https://qyapi.weixin.qq.com/cgi-bin/webhook/send\?key\=1baae34a-0c1c-4d50-9795-152c6da526b2 "hello world" \"yangxiangjun\",\"@all\"

curl ''"${HOOK}"'' \
   -H 'Content-Type: application/json' \
   -d '
   {
        "msgtype": "text",
        "text": {
            "content": "'"${MSG}"'",
             "mentioned_list":['"${MS}"'],
        }
   }'

