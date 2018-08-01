/**
 * Created by user on 26-Jun-17.
 */

app.controller('homeController', ['$http','$location', '$window', function($http,$location, $window) {
    let self = this;
    self.showTop=true;
    self.toTranslate = "";
    self.showTranslation = false;
    self.translation = "";

    self.initialmsg = function (){
        self.toTranslate = "please write a sentence for translation";
    };

    self.learn = function () {
        $http.get('http://localhost:4000/learn');
    };

    self.translate = function () {
        if (self.toTranslate != ""){
            self.showTranslation = true;
            //translateReq = "http://localhost:4000/translate_new_sentence?sentence="+ self.toTranslate;
            toTranslateJson = {"sentence" :self.toTranslate};
            $http.post('http://localhost:4000/translate_new_sentence', toTranslateJson)
                .then(function (res) {
                    self.translation =  res.data;
                    console.log(res.data);
                });
        }
    };

    self.sendmail = function () {
        subject ="translation from TextIt";
        body = "The Translation For: '"+self.toTranslate+"' \nIs: '"+self.translation+"'";
        emailAddress ="";
        $window.open("mailto:"+emailAddress+"?subject="+subject+"&body="+body,"_self");
    };

    self.getfromserver = function (){
        $http.get('http://localhost:4000/')
            .then(function (res) {
                console.log(res.data);
                $window.alert(res.data);
            });
    };



}]);
