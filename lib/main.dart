import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        backgroundColor: Colors.black54,
        body: SafeArea(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  CircleAvatar(
                    radius: 60.0,
                    backgroundImage: AssetImage('images/WhatsApp Image 2022-11-15 at 23.04.25.jpeg'),
                  ),
                  Text('LAKSHYA RAJ VIJAY',
                      style: TextStyle(
                        fontFamily: 'Aboreto',
                        fontSize: 35.0,
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      )
                  ),
                  SizedBox(
                    height: 10,
                  ),
                  Text('FLUTTER DEVELOPER',
                      style: TextStyle(
                        fontFamily: 'Coda',
                        fontSize: 25.0,
                        color: Colors.white70,
                        fontWeight: FontWeight.bold,

                      )
                  ),
                  SizedBox(
                    height: 20,
                    width: 240,
                    child: Divider(
                      color: Colors.white60,
                    ),
                  ),
                  Card(
                    margin: EdgeInsets.symmetric(vertical: 10.0, horizontal: 25.0),
                    child: ListTile(
                      leading: Icon(
                        Icons.phone,
                        color: Colors.black,
                      ),
                      title: Text(
                        '+91 xxxxxxxxxx',
                        style: TextStyle(
                          fontFamily: 'Coda',
                          fontSize: 25.0,
                          color: Colors.black87,
                        ),
                      ),
                    )),
                   Card(
                    margin: EdgeInsets.symmetric(vertical: 10.0,
                        horizontal: 25.0),
                       child: ListTile(
                         leading: Icon(
                           Icons.mail,
                           color: Colors.black,
                         ),
                         title: Text(
                           'lrvijay2003@gmail.com',
                           style: TextStyle(
                             fontFamily: 'Coda',
                             fontSize: 25.0,
                             color: Colors.black87,
                           ),
                         ),
                       )),
                    ],
              ),
            )),
      ),
    );
 }
}
