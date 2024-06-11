let user = document.getElementById("user")
alert("Hello")

let username = prompt("What is your name")

while(username.trim().length <=0){
    let username = prompt("what is your name")

    if(username.trim().length > 0){
        console.log(username)
        user.innerHTML = username
        break; }
}